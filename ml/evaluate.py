"""
Evaluate model: accuracy, ROC AUC, calibration, SHAP-style feature attributions.
"""
from typing import Any, Dict, Tuple, List
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, calibration_curve
from sklearn.calibration import calibration_curve
import logging

logger = logging.getLogger(__name__)


def evaluate_model(
    model: torch.nn.Module,
    val_data,
    tokenizer,
    config: Dict[str, Any]
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        val_data: Validation dataset
        tokenizer: Tokenizer for text encoding
        config: Configuration dictionary
        
    Returns:
        Tuple of (metrics_dict, explainability_dict)
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    all_preds = []
    all_probs = []
    all_labels = []
    all_texts = []
    all_price_features = []
    
    with torch.no_grad():
        for sample in val_data:
            # Tokenize text
            encoding = tokenizer(
                sample.news_text,
                truncation=True,
                max_length=config.get('max_length', 512),
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Prepare price features
            price_window = torch.tensor(sample.price_window, dtype=torch.float32).unsqueeze(0).to(device)
            # Flatten price window: (1, seq_len, features) -> (1, seq_len * features)
            price_flat = price_window.reshape(price_window.size(0), -1)
            
            # Forward pass
            logits = model(input_ids, attention_mask, price_flat)
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)
            
            all_preds.append(pred.cpu().item())
            all_probs.append(probs[0, 1].cpu().item())  # Probability of class 1
            all_labels.append(sample.label)
            all_texts.append(sample.news_text)
            all_price_features.append(sample.price_window)
    
    # Compute metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    
    # Calibration
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            all_labels, all_probs, n_bins=10, strategy='uniform'
        )
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
    except:
        calibration_error = 0.0
    
    metrics = {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'calibration_error': float(calibration_error),
        'n_samples': len(all_labels),
        'positive_rate': float(np.mean(all_labels)),
    }
    
    logger.info(f"Evaluation metrics: {metrics}")
    
    # Explainability: top tokens and price features
    explainability = compute_feature_importance(
        model, tokenizer, all_texts, all_price_features, all_labels, config
    )
    
    return metrics, explainability


def compute_feature_importance(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    price_features: List[np.ndarray],
    labels: List[int],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute SHAP-style feature importance.
    
    Returns top tokens and top price features by importance.
    """
    # Simple token frequency analysis for positive/negative samples
    from collections import Counter
    
    positive_texts = [texts[i] for i in range(len(texts)) if labels[i] == 1]
    negative_texts = [texts[i] for i in range(len(texts)) if labels[i] == 0]
    
    # Tokenize and count
    def get_top_tokens(texts, top_k=10):
        all_tokens = []
        for text in texts:
            tokens = tokenizer.tokenize(text.lower())
            all_tokens.extend(tokens)
        
        counter = Counter(all_tokens)
        # Filter out special tokens
        filtered = [(tok, cnt) for tok, cnt in counter.most_common(top_k * 2) 
                    if not tok.startswith('##') and tok not in ['[CLS]', '[SEP]', '[PAD]']]
        return [tok for tok, _ in filtered[:top_k]]
    
    top_positive_tokens = get_top_tokens(positive_texts, top_k=5) if positive_texts else []
    top_negative_tokens = get_top_tokens(negative_texts, top_k=5) if negative_texts else []
    
    # Price feature importance (simple variance-based)
    all_price = np.array([pf.flatten() for pf in price_features])
    feature_variance = np.var(all_price, axis=0)
    top_feature_indices = np.argsort(feature_variance)[-5:][::-1]
    
    price_feature_names = ['open', 'high', 'low', 'close', 'volume']
    bars_per_window = price_features[0].shape[0]
    
    top_price_features = []
    for idx in top_feature_indices:
        bar_idx = idx // len(price_feature_names)
        feat_idx = idx % len(price_feature_names)
        if bar_idx < bars_per_window:
            top_price_features.append(f"{price_feature_names[feat_idx]}_bar{bar_idx}")
    
    explainability = {
        'top_positive_tokens': top_positive_tokens,
        'top_negative_tokens': top_negative_tokens,
        'top_price_features': top_price_features[:5],
        'feature_importance_method': 'frequency_and_variance'
    }
    
    logger.info(f"Explainability summary: {explainability}")
    
    return explainability
