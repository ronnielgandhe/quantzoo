"""News+price hybrid regime detection strategy."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

# Try to import transformers, fall back gracefully
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


@dataclass
class RegimeHybridParams:
    """Parameters for Regime Hybrid strategy."""
    text_mode: str = "tfidf"  # "tfidf" or "hf"
    lookback: int = 20
    news_window: str = "30min"
    price_features: List[str] = None
    clf: str = "logreg"  # "logreg" or "nn"
    seed: int = 42
    contracts: int = 1
    session_start: str = "08:00"
    session_end: str = "16:30"
    tick_size: float = 0.25
    tick_value: float = 0.5
    risk_ticks: int = 100
    
    def __post_init__(self):
        if self.price_features is None:
            self.price_features = ["returns", "zscore", "atr"]


class RegimeHybrid:
    """
    News+price hybrid regime detection strategy.
    
    Combines sentiment analysis of news headlines with price-based features
    to classify market regime and trade accordingly.
    """
    
    def __init__(self, params: RegimeHybridParams):
        self.params = params
        
        # Text processing components
        self.text_vectorizer = None
        self.text_model = None
        self.text_tokenizer = None
        
        # ML components
        self.classifier = None
        self.scaler = StandardScaler()
        
        # Strategy state
        self.regime_predictions = []
        self.feature_history = []
        
        # Price feature history
        self.price_history = []
        self.returns_history = []
        self.atr_history = []
        
        np.random.seed(params.seed)
    
    def on_start(self, ctx) -> None:
        """Initialize strategy components."""
        # Initialize text processing
        if self.params.text_mode == "tfidf":
            self.text_vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
        elif self.params.text_mode == "hf" and HF_AVAILABLE:
            try:
                self.text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                self.text_model = AutoModel.from_pretrained("distilbert-base-uncased")
                self.text_model.eval()
            except Exception:
                # Fall back to TF-IDF if HF model fails
                warnings.warn("Failed to load HF model, falling back to TF-IDF")
                self.params.text_mode = "tfidf"
                self.text_vectorizer = TfidfVectorizer(
                    max_features=100,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
        else:
            # Default to TF-IDF
            self.params.text_mode = "tfidf"
            self.text_vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
        
        # Initialize classifier
        if self.params.clf == "logreg":
            self.classifier = LogisticRegression(random_state=self.params.seed)
        else:
            # Simple fallback classifier
            self.classifier = LogisticRegression(random_state=self.params.seed)
        
        # Reset state
        self.regime_predictions = []
        self.feature_history = []
        self.price_history = []
        self.returns_history = []
        self.atr_history = []
    
    def on_bar(self, ctx, bar: pd.Series) -> None:
        """Process each bar with regime detection and trading logic."""
        
        # Skip if not enough history for training
        if ctx.bar_index() < self.params.lookback * 2:
            self._update_price_history(ctx)
            return
        
        # Extract features
        features = self._extract_features(ctx, bar)
        
        if features is None:
            return
        
        # Train model on historical data (every 20 bars)
        if ctx.bar_index() % 20 == 0 and len(self.feature_history) > self.params.lookback:
            self._train_model()
        
        # Make regime prediction
        regime_pred = self._predict_regime(features)
        self.regime_predictions.append({
            'timestamp': bar.name,
            'regime': regime_pred,
            'bar_index': ctx.bar_index()
        })
        
        # Store features for future training
        self.feature_history.append(features)
        
        # Trading logic based on regime
        in_session = ctx.in_session(self.params.session_start, self.params.session_end)
        bar_confirmed = ctx.bar_confirmed()
        
        if in_session and bar_confirmed:
            current_position = ctx.position_size()
            
            # Risk-on regime: long bias
            if regime_pred > 0.5 and abs(current_position) < 1e-8:
                ctx.buy(self.params.contracts, "RegimeLong")
                self._set_exits(ctx)
            
            # Risk-off regime: flat or short bias
            elif regime_pred <= 0.5 and current_position > 1e-8:
                ctx.sell(current_position, "RegimeExit")
        
        self._update_price_history(ctx)
    
    def _extract_features(self, ctx, bar: pd.Series) -> Optional[Dict[str, Any]]:
        """Extract combined text and price features."""
        
        # Check if news data is available
        if not hasattr(bar, 'news_text') or pd.isna(bar.news_text):
            return None
        
        features = {}
        
        # Text features
        if self.params.text_mode == "tfidf" and self.text_vectorizer is not None:
            # Use simple sentiment from news aggregation
            features['news_sentiment'] = getattr(bar, 'news_sentiment', 0)
            features['news_count'] = getattr(bar, 'news_count', 0)
            features['news_sentiment_ratio'] = getattr(bar, 'news_sentiment_ratio', 1.0)
            
        elif self.params.text_mode == "hf" and self.text_model is not None:
            # Use HF embeddings (simplified for demo)
            try:
                text = str(bar.news_text)[:512]  # Limit text length
                inputs = self.text_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = self.text_model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                
                # Use first few dimensions as features
                for i in range(min(5, len(embeddings))):
                    features[f'hf_embed_{i}'] = embeddings[i]
                    
            except Exception:
                # Fallback to simple sentiment
                features['news_sentiment'] = getattr(bar, 'news_sentiment', 0)
                features['news_count'] = getattr(bar, 'news_count', 0)
        
        # Price features
        if "returns" in self.params.price_features and len(self.price_history) > 0:
            current_price = ctx.close
            prev_prices = [ctx.get_series("close", -i) for i in range(1, min(6, ctx.bar_index() + 1))]
            
            if len(prev_prices) > 0:
                returns_1 = (current_price - prev_prices[0]) / prev_prices[0] if prev_prices[0] > 0 else 0
                features['returns_1'] = returns_1
                
                if len(prev_prices) > 4:
                    returns_5 = (current_price - prev_prices[4]) / prev_prices[4] if prev_prices[4] > 0 else 0
                    features['returns_5'] = returns_5
        
        if "zscore" in self.params.price_features and len(self.returns_history) > 5:
            recent_returns = self.returns_history[-20:]
            if len(recent_returns) > 1:
                mean_ret = np.mean(recent_returns)
                std_ret = np.std(recent_returns)
                if std_ret > 0:
                    zscore = (recent_returns[-1] - mean_ret) / std_ret
                    features['returns_zscore'] = zscore
        
        if "atr" in self.params.price_features:
            # Calculate simple ATR proxy
            high = ctx.high
            low = ctx.low
            prev_close = ctx.get_series("close", -1) if ctx.bar_index() > 0 else ctx.close
            
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            features['true_range'] = tr
            
            if len(self.atr_history) > 0:
                features['atr_ratio'] = tr / (np.mean(self.atr_history[-10:]) + 1e-8)
        
        return features
    
    def _train_model(self) -> None:
        """Train the regime classification model."""
        if len(self.feature_history) < self.params.lookback:
            return
        
        try:
            # Prepare features
            features_list = []
            labels = []
            
            for i, feature_dict in enumerate(self.feature_history[:-1]):  # Exclude current bar
                if feature_dict is None:
                    continue
                
                # Create feature vector
                feature_values = []
                for key in sorted(feature_dict.keys()):
                    val = feature_dict[key]
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        feature_values.append(val)
                    else:
                        feature_values.append(0.0)
                
                if len(feature_values) > 0:
                    features_list.append(feature_values)
                    
                    # Create label based on forward return (simplified)
                    # Look ahead 1-5 bars to determine regime
                    future_return = 0
                    if i + 5 < len(self.returns_history):
                        future_return = np.mean(self.returns_history[i+1:i+6])
                    
                    # Risk-on if positive forward return
                    labels.append(1 if future_return > 0 else 0)
            
            if len(features_list) > 5 and len(set(labels)) > 1:
                X = np.array(features_list)
                y = np.array(labels)
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Train classifier
                self.classifier.fit(X_scaled, y)
                
        except Exception as e:
            # Silent fallback - use default prediction
            pass
    
    def _predict_regime(self, features: Dict[str, Any]) -> float:
        """Predict current market regime."""
        if self.classifier is None or features is None:
            return 0.5  # Neutral regime
        
        try:
            # Prepare feature vector
            feature_values = []
            for key in sorted(features.keys()):
                val = features[key]
                if isinstance(val, (int, float)) and not np.isnan(val):
                    feature_values.append(val)
                else:
                    feature_values.append(0.0)
            
            if len(feature_values) == 0:
                return 0.5
            
            # Scale and predict
            X = np.array(feature_values).reshape(1, -1)
            
            # Handle case where scaler hasn't been fitted
            try:
                X_scaled = self.scaler.transform(X)
            except:
                X_scaled = X
            
            # Get prediction probability
            try:
                prob = self.classifier.predict_proba(X_scaled)[0]
                return prob[1] if len(prob) > 1 else 0.5
            except:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _set_exits(self, ctx) -> None:
        """Set exit conditions."""
        # Simple stop loss
        stop_loss_price_offset = self.params.risk_ticks * self.params.tick_size
        ctx.set_exit(stop_loss=stop_loss_price_offset)
    
    def _update_price_history(self, ctx) -> None:
        """Update price history for feature calculation."""
        self.price_history.append(ctx.close)
        
        # Calculate returns
        if len(self.price_history) > 1:
            ret = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
            self.returns_history.append(ret)
        
        # Calculate ATR component
        if ctx.bar_index() > 0:
            high = ctx.high
            low = ctx.low
            prev_close = ctx.get_series("close", -1)
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            self.atr_history.append(tr)
        
        # Keep history manageable
        max_history = self.params.lookback * 5
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.returns_history = self.returns_history[-max_history:]
            self.atr_history = self.atr_history[-max_history:]
    
    def get_regime_predictions(self) -> List[Dict[str, Any]]:
        """Get regime predictions for reporting."""
        return self.regime_predictions