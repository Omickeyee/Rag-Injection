from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from config.settings import settings
from src.defenses.detector.dataset import create_training_dataset
from src.defenses.detector.train import train_detector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train the injection detector.")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device batch size")
    parser.add_argument("--eval-split", type=float, default=0.15, help="Fraction held out for eval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    corpus_path = settings.data_output_dir / "corpus.json"
    manifest_path = settings.data_output_dir / "manifest.json"
    if not corpus_path.exists():
        logger.error(
            "Corpus not found at %s. Run 'python scripts/generate_data.py' first.",
            corpus_path,
        )
        sys.exit(1)
    if not manifest_path.exists():
        logger.error(
            "Manifest not found at %s. Run 'python scripts/generate_data.py' first.",
            manifest_path,
        )
        sys.exit(1)
    logger.info("Creating training dataset from %s …", corpus_path)
    dataset = create_training_dataset(
        corpus_path=corpus_path,
        manifest_path=manifest_path,
        augment=True,
        seed=args.seed,
    )
    logger.info("Dataset size: %d samples", len(dataset))
    output_dir = settings.model_output_dir / "detector"
    logger.info("Training detector → %s", output_dir)
    metrics = train_detector(
        dataset=dataset,
        output_dir=output_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        eval_split=args.eval_split,
        seed=args.seed,
    )
    print("\nInjection Detector — Training Complete")
    print(f"\t- Accuracy:  {metrics['accuracy']}")
    print(f"\t- Precision: {metrics['precision']}")
    print(f"\t- Recall:    {metrics['recall']}")
    print(f"\t- F1 Score:  {metrics['f1']}")
    print(f"\t- Model saved to: {output_dir}")

if __name__ == "__main__":
    main()
