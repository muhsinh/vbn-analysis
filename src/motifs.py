"""Motif discovery from pose features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def motifs_kmeans(features: pd.DataFrame, n_clusters: int = 8) -> pd.DataFrame:
    if features is None or features.empty:
        return pd.DataFrame()
    from sklearn.cluster import KMeans

    df = features.copy()
    feature_cols = [c for c in df.columns if c != "t"]
    X = df[feature_cols].fillna(0.0).to_numpy()
    model = KMeans(n_clusters=n_clusters, random_state=0)
    labels = model.fit_predict(X)
    out = pd.DataFrame({"t": df["t"].to_numpy(), "motif_id": labels})
    return out


def motifs_hmm(features: pd.DataFrame, n_states: int = 8) -> pd.DataFrame:
    if features is None or features.empty:
        return pd.DataFrame()
    try:
        from hmmlearn import hmm
    except ImportError:
        raise ImportError("hmmlearn is required for HMM motifs")

    df = features.copy()
    feature_cols = [c for c in df.columns if c != "t"]
    X = df[feature_cols].fillna(0.0).to_numpy()
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
    states = model.fit_predict(X)
    out = pd.DataFrame({"t": df["t"].to_numpy(), "motif_id": states})
    return out
