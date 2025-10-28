"""
Generate synthetic news-price sample data for testing.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data.news_price_loader import create_synthetic_sample_data

if __name__ == "__main__":
    output_path = Path("data/examples/news_price_sample.parquet")
    create_synthetic_sample_data(output_path, n_samples=100, seed=42)
    print(f"✅ Created sample data at {output_path}")
