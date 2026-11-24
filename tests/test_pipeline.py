"""
Unit and Integration Tests for Income Prediction Pipeline
Verifies data loading, preprocessing, model serialization, inference robustness,
unseen category handling, and missing value resilience.
"""

from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import (
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    load_and_clean_data,
    split_data,
)
from src.predict import load_model, predict_income


class TestIncomePredictionPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_path = PROJECT_ROOT / "models" / "best_income_pipeline.joblib"

    def test_01_data_loading_and_target_normalization(self):
        """Test dataset loading, cleaning, and strict two-class target assertion."""
        df_clean, report = load_and_clean_data()
        self.assertGreater(len(df_clean), 0)
        self.assertIn("income", df_clean.columns)

        unique_targets = set(df_clean["income"].unique())
        self.assertEqual(unique_targets, {NEGATIVE_CLASS, POSITIVE_CLASS})

        # Check duplicates removed
        self.assertEqual(report["dropped_duplicates"], 52)
        self.assertEqual(report["final_rows"], 48790)

    def test_02_train_test_split_stratification(self):
        """Verify train-test split ratio (80:20) and class stratification."""
        df_clean, _ = load_and_clean_data()
        X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.2, random_state=42)

        total_samples = len(df_clean)
        self.assertEqual(len(X_train), 39032)
        self.assertEqual(len(X_test), 9758)

        # Proportion should be within 0.1% of each other
        train_pos_ratio = (y_train == POSITIVE_CLASS).mean()
        test_pos_ratio = (y_test == POSITIVE_CLASS).mean()
        self.assertAlmostEqual(train_pos_ratio, test_pos_ratio, places=3)

    def test_03_model_file_exists_and_loads(self):
        """Verify trained pipeline exists on disk and loads without errors."""
        self.assertTrue(self.model_path.exists(), f"Model file missing at: {self.model_path}")
        model = load_model(self.model_path)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "predict_proba"))

    def test_04_single_inference_output(self):
        """Verify prediction and probability output types and bounds for single record."""
        sample_record = {
            "age": 42.0,
            "workclass": "Private",
            "education-num": 14.0,  # Masters
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 15000.0,
            "capital-loss": 0.0,
            "hours-per-week": 50.0,
            "native-country": "United-States",
        }

        preds, probas = predict_income(sample_record, model_path=self.model_path)
        self.assertEqual(len(preds), 1)
        self.assertIn(preds[0], [NEGATIVE_CLASS, POSITIVE_CLASS])
        self.assertEqual(len(probas), 1)
        self.assertTrue(0.0 <= probas[0] <= 1.0)
        # Highly educated executive with $15k capital gain is expected to be >50K
        self.assertEqual(preds[0], POSITIVE_CLASS)

    def test_05_unseen_categories_handling(self):
        """Ensure pipeline handles unseen or rare categorical values gracefully without crashing."""
        edge_record = {
            "age": 28.0,
            "workclass": "Unknown-Sector-99",  # Unseen category
            "education-num": 10.0,
            "marital-status": "Complicated-Status",  # Unseen category
            "occupation": "Astronaut",  # Unseen category
            "relationship": "Roommate",  # Unseen category
            "race": "Martian",  # Unseen category
            "sex": "Other",  # Unseen category
            "capital-gain": 0.0,
            "capital-loss": 0.0,
            "hours-per-week": 40.0,
            "native-country": "Atlantis",  # Unseen category
        }

        # Should not raise any ValueError or KeyError
        preds, probas = predict_income(edge_record, model_path=self.model_path)
        self.assertEqual(len(preds), 1)
        self.assertIn(preds[0], [NEGATIVE_CLASS, POSITIVE_CLASS])
        self.assertTrue(0.0 <= probas[0] <= 1.0)

    def test_06_missing_values_handling(self):
        """Ensure pipeline handles missing (NaN) values in both numeric and categorical columns."""
        missing_record = {
            "age": np.nan,
            "workclass": np.nan,
            "education-num": np.nan,
            "marital-status": np.nan,
            "occupation": None,
            "relationship": None,
            "race": None,
            "sex": None,
            "capital-gain": np.nan,
            "capital-loss": np.nan,
            "hours-per-week": np.nan,
            "native-country": None,
        }

        preds, probas = predict_income(missing_record, model_path=self.model_path)
        self.assertEqual(len(preds), 1)
        self.assertIn(preds[0], [NEGATIVE_CLASS, POSITIVE_CLASS])
        self.assertTrue(0.0 <= probas[0] <= 1.0)

    def test_07_batch_inference_consistency(self):
        """Verify batch DataFrame inference produces matching array lengths."""
        batch_records = pd.DataFrame([
            {
                "age": 20.0,
                "workclass": "Private",
                "education-num": 9.0,
                "marital-status": "Never-married",
                "occupation": "Other-service",
                "relationship": "Own-child",
                "race": "White",
                "sex": "Female",
                "capital-gain": 0.0,
                "capital-loss": 0.0,
                "hours-per-week": 20.0,
                "native-country": "United-States",
            },
            {
                "age": 52.0,
                "workclass": "Self-emp-inc",
                "education-num": 16.0,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 99999.0,
                "capital-loss": 0.0,
                "hours-per-week": 60.0,
                "native-country": "United-States",
            },
        ])

        preds, probas = predict_income(batch_records, model_path=self.model_path)
        self.assertEqual(len(preds), 2)
        self.assertEqual(len(probas), 2)
        self.assertEqual(preds[0], NEGATIVE_CLASS)
        self.assertEqual(preds[1], POSITIVE_CLASS)


if __name__ == "__main__":
    unittest.main()
