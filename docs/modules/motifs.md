# motifs

Behavioral motif discovery from pose features. Assigns each time point to a
discrete behavioral state (motif) using unsupervised clustering. Two
algorithms are provided: K-Means for simple, fast clustering and a Gaussian
HMM for temporally-aware state segmentation.

**Source:** `src/motifs.py`

---

## Functions

### `motifs_kmeans`

```python
def motifs_kmeans(
    features: pd.DataFrame,
    n_clusters: int = 8,
) -> pd.DataFrame
```

Discover behavioral motifs using K-Means clustering on pose features.

Each time point is assigned to one of `n_clusters` discrete behavioral
states based on its feature vector. K-Means is fast and deterministic
(with `random_state=0`) but does not model temporal transitions.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `features` | `pd.DataFrame` | -- | Pose feature table with a `t` column and numeric feature columns (e.g., output of `derive_pose_features()`). NaN values are filled with `0.0`. |
| `n_clusters` | `int` | `8` | Number of behavioral motifs to discover. |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | Two-column DataFrame: `t` (time) and `motif_id` (integer cluster label, `0` to `n_clusters - 1`). Returns an empty DataFrame if input is `None` or empty. |

!!! tip "Choosing `n_clusters`"
    Start with 6-10 clusters and inspect the motif transition matrix
    (`viz.plot_motif_transition`) and the temporal distribution. Too few
    clusters merge distinct behaviors; too many create noisy splits.

**Example:**

```python
from motifs import motifs_kmeans
from features_pose import derive_pose_features

features = derive_pose_features(pose_df)
motif_df = motifs_kmeans(features, n_clusters=6)
print(motif_df["motif_id"].value_counts())
# 0    12345
# 1    10234
# 2     8901
# ...
```

---

### `motifs_hmm`

```python
def motifs_hmm(
    features: pd.DataFrame,
    n_states: int = 8,
) -> pd.DataFrame
```

Discover behavioral motifs using a Gaussian Hidden Markov Model (HMM).

Unlike K-Means, the HMM models temporal transitions between states, so it
produces smoother state sequences that respect the time ordering of the data.
Uses a diagonal covariance matrix and runs for 100 EM iterations.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `features` | `pd.DataFrame` | -- | Pose feature table with a `t` column and numeric feature columns. NaN values are filled with `0.0`. |
| `n_states` | `int` | `8` | Number of hidden states (behavioral motifs). |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | Two-column DataFrame: `t` and `motif_id`. Returns an empty DataFrame if input is `None` or empty. |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ImportError` | `hmmlearn` is not installed. Install with `pip install hmmlearn`. |

!!! info "HMM vs K-Means"
    - **K-Means** treats each time point independently. Good for quick
      exploration.
    - **HMM** models transition probabilities between states. Better for
      identifying genuine behavioral sequences (e.g., grooming bouts,
      locomotion episodes).

**Example:**

```python
from motifs import motifs_hmm

motif_df = motifs_hmm(features, n_states=6)
print(motif_df.head())
#            t  motif_id
# 0  0.000000         2
# 1  0.033333         2
# 2  0.066667         2
# 3  0.100000         4
# 4  0.133333         4

# Visualize transitions
from viz import plot_motif_transition
plot_motif_transition(motif_df)
```
