"""Predict classes and class probabilities from a trusted saved pipeline."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from cirrhosis_prediction.data import features


def predict(model_path: Path, input_path: Path, output_path: Path) -> pd.DataFrame:
    # joblib uses pickle: load only artifacts you created or trust.
    model = joblib.load(model_path)
    frame = pd.read_csv(input_path)
    probabilities = model.predict_proba(features(frame))
    result = pd.DataFrame(probabilities, columns=[f"prob_{c}" for c in model.classes_])
    result.insert(0, "Status", model.classes_[probabilities.argmax(axis=1)])
    if "id" in frame:
        result.insert(0, "id", frame["id"].to_numpy())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/predictions.csv"))
    args = parser.parse_args()
    predict(args.model, args.input, args.output)


if __name__ == "__main__":
    main()
