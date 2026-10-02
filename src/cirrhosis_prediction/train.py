"""Train a baseline, evaluate once on a holdout and save the full pipeline."""

import argparse
import hashlib
import json
import logging
import platform
import tomllib
from pathlib import Path

import joblib
import sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from cirrhosis_prediction.data import load_training
from cirrhosis_prediction.pipeline import build_pipeline

LOGGER = logging.getLogger(__name__)


def train(config_path: Path) -> dict:
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)["training"]
    # Paths in the checked-in config are relative to the repository root.
    root = config_path.resolve().parent.parent
    data_path = root / config["data"]
    output = root / config["output"]
    x, y = load_training(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config["test_size"],
        stratify=y,
        random_state=config["random_state"],
    )
    if y_train.value_counts().min() < config["cv_folds"]:
        raise ValueError("Each training class needs at least cv_folds observations")
    pipeline = build_pipeline(config["random_state"], config["C"], config["max_iter"])
    cv = StratifiedKFold(
        n_splits=config["cv_folds"], shuffle=True, random_state=config["random_state"]
    )
    scores = -cross_val_score(
        pipeline, x_train, y_train, cv=cv, scoring="neg_log_loss", error_score="raise"
    )
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)
    predictions = pipeline.classes_[probabilities.argmax(axis=1)]
    metrics = {
        "cv_log_loss_mean": float(scores.mean()),
        "cv_log_loss_std": float(scores.std()),
        "cv_log_loss_folds": scores.tolist(),
        "test_log_loss": float(
            log_loss(y_test, probabilities, labels=pipeline.classes_)
        ),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_f1_macro": float(f1_score(y_test, predictions, average="macro")),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "classes": pipeline.classes_.tolist(),
        "config": config,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output / "model.joblib")
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Saved model and metrics to %s", output)
    LOGGER.info("Holdout log loss: %.4f", metrics["test_log_loss"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.toml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train(args.config)


if __name__ == "__main__":
    main()
