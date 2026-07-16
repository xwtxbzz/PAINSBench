"""Model implementations for PAINSBench."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor


def train_rf(X_train, y_train, **kwargs):
    params = dict(n_estimators=50, max_depth=12, n_jobs=4,
                  random_state=42, min_samples_leaf=10, verbose=1,
                  max_samples=0.5)
    params.update(kwargs)
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def train_xgb(X_train, y_train, **kwargs):
    params = dict(n_estimators=300, max_depth=8, learning_rate=0.08,
                  subsample=0.8, colsample_bytree=0.8, n_jobs=4,
                  random_state=42, verbosity=0, tree_method="hist")
    params.update(kwargs)
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


MODEL_REGISTRY = {
    "RF": train_rf,
    "XGBoost": train_xgb,
}
