"""
Regression-based policy impact model using Gradient Boosting with 5-fold CV.
Supports year-range retraining, batch prediction, and bootstrap confidence intervals.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

from src.data_loader import get_feature_matrix, FEATURE_COLS
from src.preprocessing import prepare_pipeline, filter_by_year_range


class PolicyImpactModel:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.08, max_depth=4,
            min_samples_split=5, subsample=0.85, random_state=42)
        self.scaler        = StandardScaler()
        self.feature_names = FEATURE_COLS
        self.is_trained    = False
        self.cv_scores: Dict = {}

    def train(self, df: pd.DataFrame) -> Dict:
        df_processed = prepare_pipeline(df)
        X, y, feature_names = get_feature_matrix(df_processed)
        self.feature_names = feature_names
        X_scaled = self.scaler.fit_transform(X)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2   = cross_val_score(self.model, X_scaled, y, cv=kf, scoring="r2")
        cv_rmse = np.sqrt(-cross_val_score(
            self.model, X_scaled, y, cv=kf, scoring="neg_mean_squared_error"))
        self.model.fit(X_scaled, y)
        self.is_trained = True
        y_pred = self.model.predict(X_scaled)
        self.cv_scores = {
            "r2_mean":   round(float(cv_r2.mean()), 4),
            "r2_std":    round(float(cv_r2.std()),  4),
            "rmse_mean": round(float(cv_rmse.mean()), 4),
            "rmse_std":  round(float(cv_rmse.std()),  4),
            "train_r2":  round(float(r2_score(y, y_pred)), 4),
            "n_samples": len(y),
        }
        return self.cv_scores

    def retrain_on_range(self, df: pd.DataFrame, year_range: Tuple[int, int]) -> Dict:
        """Re-fit on a year-filtered subset; falls back to full training if < 30 rows."""
        df_filtered = filter_by_year_range(df, year_range)
        return self.train(df_filtered if len(df_filtered) >= 30 else df)

    def predict(self, feature_dict: dict) -> float:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        row = np.array([[feature_dict.get(f, 0.0) for f in self.feature_names]])
        return float(self.model.predict(self.scaler.transform(row))[0])

    def predict_batch(self, feature_dicts: List[dict]) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        matrix = np.array([[d.get(f, 0.0) for f in self.feature_names] for d in feature_dicts])
        return self.model.predict(self.scaler.transform(matrix))

    def confidence_interval(self, feature_dict, df, n_bootstrap=200, ci=0.90):
        df_processed = prepare_pipeline(df)
        X, y, _ = get_feature_matrix(df_processed)
        X_scaled   = self.scaler.transform(X)
        row_scaled = self.scaler.transform(
            np.array([[feature_dict.get(f, 0.0) for f in self.feature_names]]))
        preds = []
        rng   = np.random.default_rng(42)
        alpha = (1.0 - ci) / 2.0
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(X_scaled), size=len(X_scaled))
            m   = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
            m.fit(X_scaled[idx], y[idx])
            preds.append(float(m.predict(row_scaled)[0]))
        return round(float(np.quantile(preds, alpha)), 3), round(float(np.quantile(preds, 1-alpha)), 3)

    def feature_importance(self) -> pd.DataFrame:
        imp = self.model.feature_importances_
        return (pd.DataFrame({"feature": self.feature_names, "importance": imp})
                .sort_values("importance", ascending=False).reset_index(drop=True))
# memory optimize
# Hyperparams
# Cross validation
# Hyperparams
# Cross validation
