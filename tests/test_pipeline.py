"""Check inference contract and serialization on small synthetic data."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, cross_validate

from cirrhosis_prediction.data import CATEGORICAL, NUMERIC, features, load_training
from cirrhosis_prediction.pipeline import build_pipeline
from cirrhosis_prediction.predict import predict


@pytest.fixture
def dataset():
    rng = np.random.default_rng(8)
    frame = pd.DataFrame({name: rng.normal(size=45) for name in NUMERIC})
    for name in CATEGORICAL:
        frame[name] = np.resize(["A", "B", "A"], len(frame))
    frame.loc[0, "Age"] = np.nan
    frame["Status"] = np.resize(["C", "CL", "D"], len(frame))
    frame["id"] = np.arange(len(frame))
    return frame


def test_roundtrip_and_unseen_category(dataset, tmp_path):
    model = build_pipeline().fit(features(dataset), dataset["Status"])
    inference = dataset.drop(columns="Status").iloc[:5].copy()
    inference.loc[0, "Drug"] = "unseen drug"
    model_path = tmp_path / "model.joblib"
    input_path = tmp_path / "input.csv"
    joblib.dump(model, model_path)
    inference.to_csv(input_path, index=False)
    result = predict(model_path, input_path, tmp_path / "out.csv")
    np.testing.assert_allclose(result.filter(like="prob_").sum(axis=1), 1.0)
    np.testing.assert_allclose(
        result.filter(like="prob_"), model.predict_proba(features(inference))
    )
    assert result["id"].tolist() == inference["id"].tolist()
    assert set(result["Status"]) <= {"C", "CL", "D"}


def test_scaler_is_fitted_separately_in_each_cv_fold(dataset):
    x = features(dataset)
    y = dataset["Status"]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=8)
    result = cross_validate(build_pipeline(), x, y, cv=cv, return_estimator=True)
    for estimator, (train_indices, _) in zip(result["estimator"], cv.split(x, y)):
        numeric = estimator.named_steps["preprocessor"].named_transformers_["numeric"]
        train_numeric = x.iloc[train_indices][NUMERIC]
        expected = train_numeric.fillna(train_numeric.median()).mean().to_numpy()
        np.testing.assert_allclose(numeric.named_steps["scaler"].mean_, expected)


def test_missing_feature_rejected(dataset):
    with pytest.raises(ValueError, match="Missing feature columns"):
        features(dataset.drop(columns="Albumin"))


def test_bad_target_rejected(dataset, tmp_path):
    dataset.loc[0, "Status"] = "unknown"
    path = tmp_path / "bad.csv"
    dataset.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Status must contain"):
        load_training(path)
