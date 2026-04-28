"""
Classical ML Baseline Models — Phase 4 of wildfire-uncertainty-gnn.

WHY THESE THREE BASELINES?
---------------------------
Ridge Regression (linear):
  Sets the absolute floor. If linear model R² = -0.80 under geographic
  split, that tells us the problem is hard and non-linear methods are needed.
  Any model that cannot beat Ridge is broken.

Random Forest (tree ensemble, non-linear):
  The standard strong baseline for tabular geospatial data.
  Uses the same 61 node features as the GNN but has NO spatial structure —
  it treats each node independently, ignoring its neighbors.
  If RF beats GNN: the graph topology is not helping.
  If GNN beats RF: message-passing adds spatial value.

XGBoost (gradient boosting, non-linear):
  Often the strongest tabular baseline. Handles feature interactions
  implicitly through tree splits. Computationally efficient.
  We compare: does gradient boosting exploit feature interactions
  better than our explicit interaction features?

KEY DESIGN DECISION — NO SPATIAL CONTEXT IN BASELINES
------------------------------------------------------
RF and XGBoost see ONLY the 61 node features. They do NOT see neighbor
features or spatial position. This is intentional — it isolates the
contribution of the graph topology. If the GNN adds R² beyond these
baselines, it is because message-passing provides spatial context that
the raw features alone cannot.

TRAINING SPLIT
--------------
All baselines train on graph.train_mask nodes only.
All baselines evaluate on graph.test_mask nodes only.
The split is identical to what the GNN will use in Phase 5.
This ensures fair, apples-to-apples comparison.
"""

from __future__ import annotations
import time
import numpy as np
from typing import Optional


class RidgeBaseline:
    """
    Ridge Regression (L2-regularized linear model).

    This is the LINEAR CEILING CHECK. The geographic split makes this
    model fail catastrophically (expected R² ≈ -0.8 to -0.3) because:
    - Train BP mean ≈ 0.013 (northern Greece)
    - Test BP mean  ≈ 0.028 (southern Greece + islands)
    - Ridge predicts something close to train mean → large test error

    Any model that cannot beat Ridge is broken.
    Ridge beating other models would be a red flag — check your data.
    """

    def __init__(self, alpha: float = 1.0):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=alpha)),
        ])
        self.name = "Ridge (linear)"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeBaseline":
        t0 = time.time()
        self.model.fit(X, y.ravel())
        self.fit_time = time.time() - t0
        print(f"  ✓  Ridge fitted in {self.fit_time:.1f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class RandomForestBaseline:
    """
    Random Forest Regressor.

    PRIMARY CLASSICAL BASELINE. Handles non-linearity and feature
    interactions implicitly. Does not use spatial graph structure.

    Hyperparameters chosen for stability over tuning:
    - n_estimators=500: enough trees to reduce variance
    - max_features='sqrt': standard for regression RF
    - min_samples_leaf=10: prevents overfitting on small leaf nodes
    - n_jobs=-1: uses all CPU cores

    Feature importance: RF provides feature importance scores. We
    compare these against the Pearson correlations computed in Phase 3
    to validate that the model has learned reasonable structure.
    """

    def __init__(
        self,
        n_estimators:    int  = 500,
        max_features:    str  = "sqrt",
        min_samples_leaf:int  = 10,
        max_depth:       Optional[int] = None,
        n_jobs:          int  = -1,
        random_state:    int  = 42,
    ):
        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(
            n_estimators     = n_estimators,
            max_features     = max_features,
            min_samples_leaf = min_samples_leaf,
            max_depth        = max_depth,
            n_jobs           = n_jobs,
            random_state     = random_state,
        )
        self.name            = "Random Forest"
        self.feature_names_: Optional[list] = None

    def fit(
        self,
        X:             np.ndarray,
        y:             np.ndarray,
        feature_names: Optional[list] = None,
    ) -> "RandomForestBaseline":
        t0 = time.time()
        self.model.fit(X, y.ravel())
        self.fit_time         = time.time() - t0
        self.feature_names_   = feature_names
        print(f"  ✓  Random Forest fitted in {self.fit_time:.1f}s  "
              f"({self.model.n_estimators} trees)")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def top_feature_importances(self, n: int = 15) -> list[tuple[str, float]]:
        """Return top-n features by Gini importance."""
        importances = self.model.feature_importances_
        names = self.feature_names_ or [f"f{i}" for i in range(len(importances))]
        ranked = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
        return ranked[:n]

    def print_importances(self, n: int = 15) -> None:
        """Print top feature importances."""
        top = self.top_feature_importances(n)
        print(f"\n  Random Forest — Top {n} Feature Importances:")
        for name, imp in top:
            bar = "█" * int(imp * 500)
            print(f"    {name:<30} {imp:.4f}  {bar}")


class XGBoostBaseline:
    """
    XGBoost Gradient Boosting Regressor.

    STRONGEST EXPECTED TABULAR BASELINE. XGBoost often outperforms RF
    on tabular data, especially with interaction terms (which we have).

    Hyperparameters tuned for stability without expensive grid search:
    - n_estimators=1000, early_stopping_rounds=50: avoids overfitting
    - max_depth=6: moderate complexity
    - learning_rate=0.05: slow enough to converge
    - subsample=0.8, colsample_bytree=0.8: regularization via subsampling
    - tree_method='hist': fast histogram-based algorithm

    If XGBoost R² > 0.30 under geographic split: features are informative.
    If XGBoost R² < 0.10: the split is extremely hard or features are weak.
    """

    def __init__(
        self,
        n_estimators:          int   = 1000,
        early_stopping_rounds: int   = 50,
        max_depth:             int   = 6,
        learning_rate:         float = 0.05,
        subsample:             float = 0.8,
        colsample_bytree:      float = 0.8,
        min_child_weight:      int   = 10,
        reg_alpha:             float = 0.1,
        reg_lambda:            float = 1.0,
        n_jobs:                int   = -1,
        random_state:          int   = 42,
    ):
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("Install xgboost: pip install xgboost --break-system-packages")

        self.params = dict(
            n_estimators          = n_estimators,
            early_stopping_rounds = early_stopping_rounds,
            max_depth             = max_depth,
            learning_rate         = learning_rate,
            subsample             = subsample,
            colsample_bytree      = colsample_bytree,
            min_child_weight      = min_child_weight,
            reg_alpha             = reg_alpha,
            reg_lambda            = reg_lambda,
            tree_method           = "hist",
            n_jobs                = n_jobs,
            random_state          = random_state,
            verbosity             = 0,
        )
        self.XGBRegressor    = XGBRegressor
        self.model           = None
        self.name            = "XGBoost"
        self.feature_names_: Optional[list] = None
        self.best_iteration_ = None

    def fit(
        self,
        X_train:       np.ndarray,
        y_train:       np.ndarray,
        X_val:         np.ndarray,
        y_val:         np.ndarray,
        feature_names: Optional[list] = None,
    ) -> "XGBoostBaseline":
        """
        Fit with early stopping on validation set.

        IMPORTANT: y_train and y_val should be in TRANSFORMED space
        (near-Gaussian) for stable training. Predictions are then
        inverse-transformed before metric computation.
        """
        t0 = time.time()
        self.model = self.XGBRegressor(**self.params)
        self.model.fit(
            X_train, y_train.ravel(),
            eval_set=[(X_val, y_val.ravel())],
            verbose=False,
        )
        self.fit_time       = time.time() - t0
        self.feature_names_ = feature_names
        self.best_iteration_ = self.model.best_iteration

        print(f"  ✓  XGBoost fitted in {self.fit_time:.1f}s  "
              f"(best iter={self.best_iteration_})")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def top_feature_importances(self, n: int = 15) -> list[tuple[str, float]]:
        """Return top-n features by XGBoost gain importance."""
        importances = self.model.feature_importances_
        names = self.feature_names_ or [f"f{i}" for i in range(len(importances))]
        ranked = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
        return ranked[:n]

    def print_importances(self, n: int = 15) -> None:
        top = self.top_feature_importances(n)
        print(f"\n  XGBoost — Top {n} Feature Importances (gain):")
        for name, imp in top:
            bar = "█" * int(imp * 300)
            print(f"    {name:<30} {imp:.4f}  {bar}")


class MeanBaseline:
    """
    Naive mean predictor — predicts training mean for every test node.

    This is the ABSOLUTE FLOOR. Any real model must beat this.
    Under geographic split, this gives R² = large negative number because
    test BP distribution ≠ train BP distribution.
    Reporting this confirms the split is genuinely hard.
    """

    def __init__(self):
        self.train_mean_ = None
        self.name        = "Naive Mean"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanBaseline":
        self.train_mean_ = float(np.mean(y))
        print(f"  ✓  Naive Mean: train mean = {self.train_mean_:.5f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.train_mean_, dtype=np.float32)