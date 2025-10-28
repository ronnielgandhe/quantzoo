"""
Train a transformer-based hybrid model for news and price sequences.
Accepts YAML config, logs model, metrics, tokenizer, and model card metadata.
"""
import argparse
import yaml
import os
import json
from typing import Any, Dict
from pathlib import Path
from datetime import datetime
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from ml.models.hybrid_transformer import HybridTransformerClassifier
from ml.data.news_price_loader import load_news_price_data, NewsPriceSample
from ml.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CARD_FIELDS = [
    "dataset", "training_data_provenance", "preprocessing_steps", "metrics",
    "evaluation_dates", "intended_use", "caveats", "license", "explainability_summary"
]


def collate_fn(batch, tokenizer, config):
    """Collate batch of samples into tensors."""
    texts = [sample.news_text for sample in batch]
    labels = torch.tensor([sample.label for sample in batch], dtype=torch.long)
    
    # Tokenize texts
    encoding = tokenizer(
        texts,
        truncation=True,
        max_length=config.get('max_length', 512),
        padding='max_length',
        return_tensors='pt'
    )
    
    # Stack price features and flatten
    price_windows = [torch.tensor(sample.price_window, dtype=torch.float32) for sample in batch]
    price_stacked = torch.stack(price_windows)  # (batch, seq_len, features)
    price_flat = price_stacked.reshape(price_stacked.size(0), -1)  # (batch, seq_len * features)
    
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'price_features': price_flat,
        'labels': labels
    }


def train_epoch(model, dataloader, optimizer, scheduler, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        price_features = batch['price_features'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask, price_features)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('max_grad_norm', 1.0))
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def save_model_card(model_dir: str, card: Dict[str, Any]) -> None:
    """Save model card as JSON and Markdown."""
    # JSON format
    with open(os.path.join(model_dir, "model_card.json"), "w") as f:
        json.dump(card, f, indent=2)
    
    # Markdown format
    md_content = f"""# Model Card: {card.get('experiment_name', 'Untitled')}

## Model Details
- **Date**: {card.get('evaluation_dates', 'N/A')}
- **License**: {card.get('license', 'Not specified')}

## Intended Use
{card.get('intended_use', 'Not specified')}

## Dataset
{card.get('dataset', 'Not specified')}

## Training Data Provenance
{card.get('training_data_provenance', 'Not specified')}

## Preprocessing Steps
{card.get('preprocessing_steps', 'Not specified')}

## Metrics
```json
{json.dumps(card.get('metrics', {}), indent=2)}
```

## Explainability Summary
```json
{json.dumps(card.get('explainability_summary', {}), indent=2)}
```

## Caveats and Limitations
{card.get('caveats', 'Not specified')}
"""
    
    with open(os.path.join(model_dir, "model_card.md"), "w") as f:
        f.write(md_content)


def main():
    parser = argparse.ArgumentParser(description="Train hybrid transformer model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Config loaded: {config}")
    
    # Set seed for reproducibility
    torch.manual_seed(config.get('seed', 42))
    
    # Load data
    logger.info(f"Loading data from {config['data_path']}...")
    train_data, val_data = load_news_price_data(config["data_path"], config)
    logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    # Tokenizer and base model
    transformer_name = config.get('transformer_model', 'distilbert-base-uncased')
    logger.info(f"Loading transformer: {transformer_name}")
    tokenizer = AutoTokenizer.from_pretrained(transformer_name)
    base_model = AutoModel.from_pretrained(transformer_name)
    
    # Calculate price feature dimension
    sample = train_data[0]
    price_feature_dim = sample.price_window.shape[0] * sample.price_window.shape[1]
    config['price_feature_dim'] = price_feature_dim
    logger.info(f"Price feature dim: {price_feature_dim}")
    
    # Create model
    model = HybridTransformerClassifier(base_model, config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"Model created on device: {device}")
    
    if args.dry_run:
        logger.info("Dry run completed successfully!")
        return
    
    # Create dataloaders
    batch_size = config.get('batch_size', 16)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, config)
    )
    
    # Optimizer and scheduler
    num_epochs = config.get('num_epochs', 3)
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(num_training_steps * config.get('warmup_ratio', 0.1))
    
    optimizer = AdamW(model.parameters(), lr=config.get('learning_rate', 2e-5))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, config)
        logger.info(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")
    
    # Evaluation
    logger.info("\nEvaluating model...")
    metrics, explainability = evaluate_model(model, val_data, tokenizer, config)
    
    # Save model, tokenizer, metrics, model card
    model_dir = os.path.join("artifacts/models", config["experiment_name"])
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model state
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': metrics,
    }, os.path.join(model_dir, "checkpoint.pt"))
    
    # Save tokenizer
    tokenizer.save_pretrained(model_dir)
    
    # Save metrics
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Build and save model card
    card = {
        'experiment_name': config['experiment_name'],
        'dataset': config.get('dataset', config['data_path']),
        'training_data_provenance': config.get('training_data_provenance', 'Synthetic data generated for testing'),
        'preprocessing_steps': config.get('preprocessing_steps', 'Tokenization, price normalization, time-based split'),
        'metrics': metrics,
        'evaluation_dates': datetime.now().isoformat(),
        'intended_use': config.get('intended_use', 'Research and development only. Not for production trading.'),
        'caveats': config.get('caveats', 'Model trained on limited synthetic data. Requires extensive validation before real-world use.'),
        'license': config.get('license', 'MIT'),
        'explainability_summary': explainability,
        'transformer_model': transformer_name,
        'training_config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': config.get('learning_rate', 2e-5),
        }
    }
    
    save_model_card(model_dir, card)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training completed!")
    logger.info(f"Model, metrics, and card saved to: {model_dir}")
    logger.info(f"Final metrics: {metrics}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
