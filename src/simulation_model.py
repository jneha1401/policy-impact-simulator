"""
Regression-based policy impact model.
Uses Gradient Boosting with 5-fold cross-validation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.data_loader import get_feature_matrix, FEATURE_COLS
from src.preprocessing import prepare_pipeline


class PolicyImpactModel:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=4,
            min_samples_split=5,
            subsample=0.85,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.feature_names = FEATURE_COLS
        self.is_trained = False
        self.cv_scores = {}

    def train(self, df: pd.DataFrame):
        """Train the model and compute CV metrics."""
        df_processed = prepare_pipeline(df)
        X, y, feature_names = get_feature_matrix(df_processed)
        self.feature_names = feature_names

        X_scaled = self.scaler.fit_transform(X)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2 = cross_val_score(self.model, X_scaled, y, cv=kf, scoring="r2")
        cv_rmse = np.sqrt(
            -cross_val_score(self.model, X_scaled, y, cv=kf, scoring="neg_mean_squared_error")
        )

        self.model.fit(X_scaled, y)
        self.is_trained = True

        y_pred = self.model.predict(X_scaled)
        self.cv_scores = {
            "r2_mean": round(float(cv_r2.mean()), 4),
            "r2_std": round(float(cv_r2.std()), 4),
            "rmse_mean": round(float(cv_rmse.mean()), 4),
            "rmse_std": round(float(cv_rmse.std()), 4),
            "train_r2": round(float(r2_score(y, y_pred)), 4),
            "n_samples": len(y),
        }
        return self.cv_scores

    def predict(self, feature_dict: dict) -> float:
        """Predict poverty rate given a dict of feature values."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        row = np.array([[feature_dict.get(f, 0) for f in self.feature_names]])
        row_scaled = self.scaler.transform(row)
        return float(self.model.predict(row_scaled)[0])

    def feature_importance(self) -> pd.DataFrame:
        """Return sorted feature importance DataFrame."""
        imp = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
