"""
Module: predict.py
Provides prediction interface for new observations using the trained pipeline.
Supports single dictionary input, batch list of dictionaries, or pandas DataFrame.
Can also be executed from the command line.
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from src.data_preprocessing import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
)

DEFAULT_MODEL_PATH = Path("models/best_income_pipeline.joblib")


def load_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
    """Load the serialized joblib pipeline model."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(path)


def format_input(data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
    """Format diverse input types into a standardized DataFrame with expected columns."""
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        raise TypeError(f"Unsupported data format: {type(data)}. Expected DataFrame, dict, or list of dicts.")

    # Clean whitespace on string columns
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("str"):
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["?", "nan", "None"]), col] = np.nan

    # Check for presence of required columns; fill missing with NaN so pipeline imputer handles them
    all_expected = DEFAULT_NUMERIC_FEATURES + DEFAULT_CATEGORICAL_FEATURES
    for col in all_expected:
        if col not in df.columns:
            df[col] = np.nan

    return df[all_expected]


def predict_income(
    data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> Tuple[List[str], np.ndarray]:
    """
    Predict income category and positive probability for given input data.
    
    Returns:
    - predictions: List of predicted labels ('<=50K' or '>50K')
    - probabilities: Array of predicted probabilities for positive class '>50K'
    """
    model = load_model(model_path)
    df_formatted = format_input(data)

    predictions = model.predict(df_formatted).tolist()

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(df_formatted)[:, 1]
    else:
        # Fallback if model has no probabilities
        probas = np.array([1.0 if p == POSITIVE_CLASS else 0.0 for p in predictions])

    return predictions, probas


def main():
    parser = argparse.ArgumentParser(description="Predict annual income (>50K vs <=50K) from census attributes.")
    parser.add_argument("--age", type=float, default=35.0, help="Age in years")
    parser.add_argument("--workclass", type=str, default="Private", help="Employment sector")
    parser.add_argument("--education-num", type=float, default=13.0, help="Years/level of education (1-16)")
    parser.add_argument("--marital-status", type=str, default="Married-civ-spouse", help="Marital status")
    parser.add_argument("--occupation", type=str, default="Exec-managerial", help="Job occupation")
    parser.add_argument("--relationship", type=str, default="Husband", help="Household relationship")
    parser.add_argument("--race", type=str, default="White", help="Race")
    parser.add_argument("--sex", type=str, default="Male", help="Sex (Male/Female)")
    parser.add_argument("--capital-gain", type=float, default=5000.0, help="Capital gains in USD")
    parser.add_argument("--capital-loss", type=float, default=0.0, help="Capital losses in USD")
    parser.add_argument("--hours-per-week", type=float, default=45.0, help="Working hours per week")
    parser.add_argument("--native-country", type=str, default="United-States", help="Native country")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to pipeline model")

    args = parser.parse_args()

    input_dict = {
        "age": args.age,
        "workclass": args.workclass,
        "education-num": args.education_num,
        "marital-status": args.marital_status,
        "occupation": args.occupation,
        "relationship": args.relationship,
        "race": args.race,
        "sex": args.sex,
        "capital-gain": args.capital_gain,
        "capital-loss": args.capital_loss,
        "hours-per-week": args.hours_per_week,
        "native-country": args.native_country,
    }

    preds, probas = predict_income(input_dict, model_path=args.model_path)
    print("\n--- Model Inference Result ---")
    print(f"Input features: {input_dict}")
    print(f"Predicted Class: {preds[0]}")
    print(f"Predicted Probability of >50K: {probas[0]:.4f} ({probas[0]*100:.2f}%)")
    print("Academic disclaimer: This inference is an academic demonstration on 1994 census data.")


if __name__ == "__main__":
    main()
