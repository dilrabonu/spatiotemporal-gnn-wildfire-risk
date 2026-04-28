"""
Classical ML Baseline Models — Phase 4.

WHY BASELINES ARE NON-NEGOTIABLE FOR PUBLICATION
-------------------------------------------------
Every reviewer will ask: "Why is your GNN better than RF or CNN?"
Without answering this with numbers, the paper cannot be published.

These 4 baselines serve two scientific purposes:
  1. Floor setting — show the problem difficulty under geographic split
  2. Ablation — isolate what graph topology adds beyond raw features

These models have NO spatial message-passing, NO neighbor awareness,
NO graph structure. Each node is predicted INDEPENDENTLY from its 61
features. If GNN beats these → graph topology adds real predictive value.

CONFIRMED SPLIT (Phase 3 fix applied)
--------------------------------------
  Train: 237,304 nodes (72.5%)  rows 0–4200
  Val:    32,570 nodes  (9.9%)  rows 4201–4800
  Test:   57,531 nodes (17.6%)  rows 4801–7590
"""
from __future__ import annotations
import time
import numpy as np
from typing import Optional


class NaiveMeanBaseline:
    """Predict training mean for every test node. Absolute floor."""
    def __init__(self):
        self.train_mean_ = None
        self.name = "Naive Mean"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveMeanBaseline":
        self.train_mean_ = float(np.mean(y))
        print(f"    Train mean = {self.train_mean_:.5f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.train_mean_, dtype=np.float32)


class RidgeBaseline:
    """Ridge Regression — linear ceiling check."""
    def __init__(self, alpha: float = 1.0):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=alpha)),
        ])
        self.name = "Ridge Regression"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeBaseline":
        t0 = time.time()
        self.model.fit(X, y.ravel())
        print(f"    Fitted in {time.time()-t0:.1f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class RandomForestBaseline:
    """
    Random Forest — primary non-linear tabular baseline.
    Uses all 61 node features, NO spatial graph structure.
    Feature importances used in paper discussion.
    """
    def __init__(self, n_estimators=500, max_features="sqrt",
                 min_samples_leaf=10, n_jobs=-1, random_state=42):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_features=max_features,
            min_samples_leaf=min_samples_leaf, n_jobs=n_jobs,
            random_state=random_state,
        )
        self.name = "Random Forest"
        self.feature_names_: Optional[list] = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[list] = None) -> "RandomForestBaseline":
        t0 = time.time()
        self.model.fit(X, y.ravel())
        self.feature_names_ = feature_names
        print(f"    Fitted {self.model.n_estimators} trees in {time.time()-t0:.1f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def top_importances(self, n: int = 15) -> list[tuple[str, float]]:
        imp   = self.model.feature_importances_
        names = self.feature_names_ or [f"f{i}" for i in range(len(imp))]
        return sorted(zip(names, imp), key=lambda x: x[1], reverse=True)[:n]

    def print_importances(self, n: int = 15) -> None:
        print(f"\n  Random Forest — Top {n} Feature Importances:")
        for name, v in self.top_importances(n):
            print(f"    {name:<30} {v:.4f}  {'█'*int(v*300)}")


class XGBoostBaseline:
    """
    XGBoost — strongest expected tabular baseline.
    Early stopping on validation set prevents overfitting.
    Trains in transformed y space; predictions inverse-transformed.
    """
    def __init__(self, n_estimators=1000, early_stopping_rounds=50,
                 max_depth=6, learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.8, min_child_weight=10,
                 reg_alpha=0.1, reg_lambda=1.0,
                 n_jobs=-1, random_state=42):
        from xgboost import XGBRegressor
        self.params = dict(
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            max_depth=max_depth, learning_rate=learning_rate,
            subsample=subsample, colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda,
            tree_method="hist", n_jobs=n_jobs,
            random_state=random_state, verbosity=0,
        )
        self.XGBRegressor = XGBRegressor
        self.model = None
        self.name  = "XGBoost"
        self.feature_names_: Optional[list] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            feature_names: Optional[list] = None) -> "XGBoostBaseline":
        t0 = time.time()
        self.model = self.XGBRegressor(**self.params)
        self.model.fit(X_train, y_train.ravel(),
                       eval_set=[(X_val, y_val.ravel())], verbose=False)
        self.feature_names_ = feature_names
        print(f"    Fitted in {time.time()-t0:.1f}s  "
              f"best_iter={self.model.best_iteration}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def top_importances(self, n: int = 15) -> list[tuple[str, float]]:
        imp   = self.model.feature_importances_
        names = self.feature_names_ or [f"f{i}" for i in range(len(imp))]
        return sorted(zip(names, imp), key=lambda x: x[1], reverse=True)[:n]

    def print_importances(self, n: int = 15) -> None:
        print(f"\n  XGBoost — Top {n} Feature Importances:")
        for name, v in self.top_importances(n):
            print(f"    {name:<30} {v:.4f}  {'█'*int(v*200)}")