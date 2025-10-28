"""
Hybrid model: transformer encoder for text + CNN/MLP for price, fused for classification.

Combines Hugging Face transformer for news text with price sequence encoder.
"""
import torch
import torch.nn as nn
from typing import Any, Dict, Optional


class HybridTransformerClassifier(nn.Module):
    """
    Hybrid classifier combining text transformer and price encoder.
    
    Architecture:
    - Text branch: Pre-trained transformer (e.g., BERT, DistilBERT)
    - Price branch: MLP or 1D CNN over price window
    - Fusion: Concatenate embeddings and pass through classifier head
    """
    
    def __init__(self, transformer, config: Dict[str, Any]):
        """
        Initialize hybrid model.
        
        Args:
            transformer: Pre-trained transformer model (from Hugging Face)
            config: Configuration dict with keys:
                - price_feature_dim: Flattened dimension of price features
                - price_hidden_dim: Hidden dimension for price encoder
                - dropout: Dropout rate
                - num_classes: Number of output classes (default 2)
        """
        super().__init__()
        self.transformer = transformer
        self.config = config
        
        # Freeze transformer initially (can be unfrozen for fine-tuning)
        if config.get('freeze_transformer', True):
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Price encoder: MLP over flattened price window
        price_input_dim = config['price_feature_dim']
        price_hidden = config.get('price_hidden_dim', 64)
        
        self.price_encoder = nn.Sequential(
            nn.Linear(price_input_dim, price_hidden),
            nn.LayerNorm(price_hidden),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(price_hidden, price_hidden // 2),
            nn.LayerNorm(price_hidden // 2),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
        )
        
        # Fusion and classifier
        transformer_dim = transformer.config.hidden_size
        fused_dim = transformer_dim + price_hidden // 2
        num_classes = config.get('num_classes', 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.LayerNorm(fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(fused_dim // 2, num_classes)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        price_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Tokenized text (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            price_features: Flattened price features (batch_size, price_feature_dim)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Text encoding: use [CLS] token embedding
        transformer_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embedding = transformer_output.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Price encoding
        price_embedding = self.price_encoder(price_features)
        
        # Fusion
        fused = torch.cat([text_embedding, price_embedding], dim=1)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits
    
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        price_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get intermediate embeddings for analysis.
        
        Returns dict with 'text', 'price', and 'fused' embeddings.
        """
        with torch.no_grad():
            transformer_output = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            text_embedding = transformer_output.last_hidden_state[:, 0, :]
            price_embedding = self.price_encoder(price_features)
            fused = torch.cat([text_embedding, price_embedding], dim=1)
            
            return {
                'text': text_embedding,
                'price': price_embedding,
                'fused': fused
            }
