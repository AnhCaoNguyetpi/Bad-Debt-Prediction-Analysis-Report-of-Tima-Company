"""Small, data-free tests for the training contract."""
import unittest
import numpy as np
import pandas as pd

from train_model import (add_safe_features, canonical, coerce_target,
                         select_features, split_indices)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        size = 200
        target = np.tile([0, 0, 0, 1], size // 4)
        rng.shuffle(target)
        self.data = pd.DataFrame({
            "CustomerID": [f"C{i // 2:03d}" for i in range(size)],
            "HasBadDebt": target,
            "HasLatePayment": target,
            "LongestOverdue": target * 30,
            "SoTienDKVayBanDau": rng.integers(1_000_000, 20_000_000, size),
            "Salary": rng.integers(3_000_000, 30_000_000, size),
            "Birthday": ["1990-01-01"] * size,
            "JobName": rng.choice(["Office", "Sales", "Other"], size),
        })

    def test_column_matching_is_accent_insensitive(self):
        self.assertEqual(canonical("Trạng thái"), canonical("Trang thai"))

    def test_target_and_safe_features(self):
        data, target = coerce_target(self.data, "has bad debt")
        featured = add_safe_features(data, "2019-03-17")
        self.assertEqual(target, "HasBadDebt")
        self.assertTrue(featured["Age"].between(18, 100).all())
        self.assertIn("LoanToIncome", featured)

    def test_leakage_and_identifiers_are_excluded(self):
        featured = add_safe_features(self.data, "2019-03-17")
        features, excluded = select_features(
            featured, "HasBadDebt", "CustomerID", False, []
        )
        self.assertNotIn("CustomerID", features)
        self.assertNotIn("HasLatePayment", features)
        self.assertNotIn("LongestOverdue", features)
        self.assertIn("HasLatePayment", excluded)

    def test_group_split_has_no_customer_overlap(self):
        y = self.data["HasBadDebt"]
        train, validation, test, strategy = split_indices(self.data, y, "CustomerID")
        groups = self.data["CustomerID"]
        self.assertTrue(set(groups.iloc[train]).isdisjoint(groups.iloc[validation]))
        self.assertTrue(set(groups.iloc[train]).isdisjoint(groups.iloc[test]))
        self.assertTrue(set(groups.iloc[validation]).isdisjoint(groups.iloc[test]))
        self.assertEqual(strategy, "customer_grouped_60_20_20")


if __name__ == "__main__":
    unittest.main()
