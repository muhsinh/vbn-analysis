"""Modeling utilities for neural-behavior fusion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from timebase import build_time_grid, bin_spike_times, bin_continuous_features


@dataclass
class DesignMatrixBuilder:
    categorical_cols: List[str]
    model_name: str
    feature_columns_: List[str] | None = None

    def fit(self, X: pd.DataFrame) -> "DesignMatrixBuilder":
        X_proc = prepare_features(X, self.categorical_cols)
        X_enc = encode_for_model(X_proc, self.categorical_cols, self.model_name)
        self.feature_columns_ = list(X_enc.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_proc = prepare_features(X, self.categorical_cols)
        X_enc = encode_for_model(X_proc, self.categorical_cols, self.model_name)
        if self.feature_columns_ is None:
            self.feature_columns_ = list(X_enc.columns)
            return X_enc
        return X_enc.reindex(columns=self.feature_columns_, fill_value=0.0)


def make_model(name: str, task: str, **kwargs) -> Any:
    name = name.lower()
    if name == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("xgboost is required for the selected model") from exc
        params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}
        params.update(kwargs)
        if task == "count":
            return xgb.XGBRegressor(objective="count:poisson", **params)
        return xgb.XGBRegressor(objective="reg:squarederror", **params)

    if name == "catboost":
        raise NotImplementedError("CatBoost not yet implemented; add here with minimal changes")

    raise ValueError(f"Unsupported model: {name}")


def prepare_features(X: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    X = X.copy()
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype(str).fillna("__MISSING__")
    # Keep numeric NaNs as-is
    return X


def encode_for_model(X: pd.DataFrame, categorical_cols: List[str], model_name: str) -> pd.DataFrame:
    if model_name == "xgboost":
        cat_cols = [c for c in categorical_cols if c in X.columns]
        if cat_cols:
            return pd.get_dummies(X, columns=cat_cols, dummy_na=False)
    return X


def time_blocked_splits(n_samples: int, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_samples < 4:
        return []
    block = max(1, n_samples // (n_splits + 1))
    splits = []
    for i in range(1, n_splits + 1):
        test_start = i * block
        test_end = min((i + 1) * block, n_samples)
        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits


def circular_shift(arr: np.ndarray, shift: int) -> np.ndarray:
    if len(arr) == 0:
        return arr
    shift = shift % len(arr)
    return np.concatenate([arr[-shift:], arr[:-shift]])


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["r2"] = float(r2_score(y_true, y_pred))
    if task == "count":
        # Pseudo-R2 based on deviance ratio (approx)
        mean_rate = np.mean(y_true) + 1e-6
        null_pred = np.full_like(y_true, mean_rate)
        dev_model = np.sum((y_true - y_pred) ** 2)
        dev_null = np.sum((y_true - null_pred) ** 2)
        metrics["pseudo_r2"] = float(1.0 - dev_model / (dev_null + 1e-9))
    return metrics


def fit_and_evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    task: str,
    categorical_cols: List[str],
    n_splits: int = 5,
) -> Dict[str, Any]:
    builder = DesignMatrixBuilder(categorical_cols=categorical_cols, model_name=model_name)
    builder.fit(X)
    splits = time_blocked_splits(len(X), n_splits=n_splits)
    if not splits:
        splits = [(np.arange(len(X)), np.arange(len(X)))]

    metrics_list = []
    for train_idx, test_idx in splits:
        X_train = builder.transform(X.iloc[train_idx])
        X_test = builder.transform(X.iloc[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]
        model = make_model(model_name, task)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics_list.append(evaluate_model(y_test, pred, task))

    avg_metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0].keys()}
    final_model = make_model(model_name, task)
    final_model.fit(builder.transform(X), y)

    return {
        "model": final_model,
        "metrics": avg_metrics,
        "feature_columns": builder.feature_columns_,
    }


def build_fusion_table(
    spike_times: Dict[str, np.ndarray] | None,
    motifs: pd.DataFrame | None,
    bin_size_s: float,
) -> pd.DataFrame:
    if spike_times:
        all_times = np.concatenate(list(spike_times.values()))
        t_start, t_end = float(all_times.min()), float(all_times.max())
    elif motifs is not None and not motifs.empty:
        t_start, t_end = float(motifs["t"].min()), float(motifs["t"].max())
    else:
        t_start, t_end = 0.0, 10.0

    time_grid = build_time_grid(t_start, t_end, bin_size_s)
    spike_counts = bin_spike_times(spike_times or {}, time_grid, bin_size_s)

    if motifs is not None and not motifs.empty:
        motifs_binned = bin_continuous_features(motifs, time_grid)
    else:
        motifs_binned = pd.DataFrame({"t": time_grid})

    fusion = spike_counts.merge(motifs_binned, on="t", how="left")
    return fusion
