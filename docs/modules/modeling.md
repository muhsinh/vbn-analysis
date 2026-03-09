# modeling

Design matrix construction, model fitting, evaluation, and neural-behavior
fusion table assembly. Provides the infrastructure for the final predictive
modeling stage of the pipeline.

**Source:** `src/modeling.py`

---

## Classes

### `DesignMatrixBuilder`

```python
@dataclass
class DesignMatrixBuilder:
    categorical_cols: List[str]
    model_name: str
    feature_columns_: List[str] | None = None
```

Scikit-learn-style transformer that prepares raw feature DataFrames into
model-ready design matrices. Handles categorical encoding and ensures
consistent column ordering between training and inference.

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `categorical_cols` | `List[str]` | Column names to treat as categorical variables. |
| `model_name` | `str` | Model name (e.g., `"xgboost"`). Determines the encoding strategy. |
| `feature_columns_` | `List[str] \| None` | Learned column order after `fit()`. `None` until fitted. |

#### Methods

##### `fit(X)`

```python
def fit(self, X: pd.DataFrame) -> DesignMatrixBuilder
```

Learn the column structure from a training DataFrame.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `pd.DataFrame` | Training features. |

**Returns:** `self` (for method chaining).

##### `transform(X)`

```python
def transform(self, X: pd.DataFrame) -> pd.DataFrame
```

Transform a DataFrame into the design matrix format learned during `fit()`.
Missing columns are filled with `0.0`; extra columns are dropped.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `pd.DataFrame` | Features to transform. |

**Returns:** `pd.DataFrame` with columns matching `feature_columns_`.

**Example:**

```python
from modeling import DesignMatrixBuilder

builder = DesignMatrixBuilder(
    categorical_cols=["trial_type", "motif_id"],
    model_name="xgboost",
)
builder.fit(train_df)

X_train = builder.transform(train_df)
X_test = builder.transform(test_df)
# X_test has exactly the same columns as X_train
```

---

## Functions

### `make_model`

```python
def make_model(
    name: str,
    task: str,
    **kwargs,
) -> Any
```

Instantiate a model by name and task type.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | -- | Model name (case-insensitive). Currently supported: `"xgboost"`. |
| `task` | `str` | -- | Task type. `"count"` uses Poisson objective (for spike counts); any other value uses squared-error regression. |
| `**kwargs` | -- | -- | Override default hyperparameters (e.g., `n_estimators=500`). |

**Returns:**

| Type | Description |
|------|-------------|
| `Any` | A scikit-learn-compatible estimator. |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ImportError` | `xgboost` is not installed. |
| `NotImplementedError` | `name="catboost"` (planned but not yet implemented). |
| `ValueError` | Unsupported model name. |

**Default XGBoost hyperparameters:**

| Parameter | Value |
|-----------|-------|
| `n_estimators` | `200` |
| `max_depth` | `4` |
| `learning_rate` | `0.05` |

**Example:**

```python
from modeling import make_model

# Poisson model for spike counts
model = make_model("xgboost", task="count")

# Regression model with custom params
model = make_model("xgboost", task="regression", n_estimators=500, max_depth=6)
```

---

### `prepare_features`

```python
def prepare_features(
    X: pd.DataFrame,
    categorical_cols: List[str],
) -> pd.DataFrame
```

Prepare a feature DataFrame for modeling by converting categorical columns
to strings and filling missing categorical values with `"__MISSING__"`.
Numeric NaNs are left as-is (XGBoost handles them natively).

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `pd.DataFrame` | -- | Raw features. |
| `categorical_cols` | `List[str]` | -- | Columns to treat as categorical. |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | Copy of `X` with categorical columns converted to strings. |

**Example:**

```python
from modeling import prepare_features

X_clean = prepare_features(df, categorical_cols=["trial_type", "motif_id"])
print(X_clean["trial_type"].dtype)  # object (string)
```

---

### `encode_for_model`

```python
def encode_for_model(
    X: pd.DataFrame,
    categorical_cols: List[str],
    model_name: str,
) -> pd.DataFrame
```

Encode categorical columns based on the target model. For XGBoost, applies
one-hot encoding via `pd.get_dummies()`.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `pd.DataFrame` | -- | Prepared feature DataFrame. |
| `categorical_cols` | `List[str]` | -- | Columns to encode. |
| `model_name` | `str` | -- | Model name (determines encoding strategy). |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | Encoded DataFrame. For `"xgboost"`, categorical columns are replaced by one-hot indicator columns. |

**Example:**

```python
from modeling import encode_for_model

X_enc = encode_for_model(X_prepared, ["trial_type"], "xgboost")
# "trial_type" column is replaced by "trial_type_go", "trial_type_no-go", etc.
```

---

### `time_blocked_splits`

```python
def time_blocked_splits(
    n_samples: int,
    n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]
```

Generate time-blocked train/test splits for temporal data. Unlike random
K-Fold, this respects the temporal ordering of the data by using contiguous
blocks as test sets and all preceding data as training data.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_samples` | `int` | -- | Total number of samples. |
| `n_splits` | `int` | `5` | Number of folds. |

**Returns:**

| Type | Description |
|------|-------------|
| `List[Tuple[np.ndarray, np.ndarray]]` | List of `(train_indices, test_indices)` tuples. May be empty if `n_samples < 4`. |

!!! warning "Not standard K-Fold"
    Each split uses all data before the test block for training, and the test
    block is a contiguous time segment. This prevents temporal leakage.

**Example:**

```python
from modeling import time_blocked_splits

splits = time_blocked_splits(1000, n_splits=5)
for i, (train, test) in enumerate(splits):
    print(f"Fold {i}: train={len(train)} test={len(test)}")
# Fold 0: train=167 test=167
# Fold 1: train=334 test=167
# ...
```

---

### `evaluate_model`

```python
def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
) -> Dict[str, float]
```

Compute evaluation metrics for a fitted model.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_true` | `np.ndarray` | -- | Ground truth values. |
| `y_pred` | `np.ndarray` | -- | Model predictions. |
| `task` | `str` | -- | Task type. `"count"` additionally computes a pseudo-R-squared based on deviance ratio. |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, float]` | Metrics dictionary. Always contains `r2`. Contains `pseudo_r2` if `task="count"`. |

**Example:**

```python
from modeling import evaluate_model

metrics = evaluate_model(y_true, y_pred, task="count")
print(f"R2={metrics['r2']:.3f}, Pseudo-R2={metrics['pseudo_r2']:.3f}")
```

---

### `fit_and_evaluate`

```python
def fit_and_evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    task: str,
    categorical_cols: List[str],
    n_splits: int = 5,
) -> Dict[str, Any]
```

End-to-end model fitting with cross-validation. Builds the design matrix,
runs time-blocked CV, and fits a final model on all data.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `pd.DataFrame` | -- | Raw feature DataFrame. |
| `y` | `np.ndarray` | -- | Target array. |
| `model_name` | `str` | -- | Model name (e.g., `"xgboost"`). |
| `task` | `str` | -- | `"count"` or `"regression"`. |
| `categorical_cols` | `List[str]` | -- | Categorical column names. |
| `n_splits` | `int` | `5` | Number of CV folds. |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Result dictionary: |

| Key | Type | Description |
|-----|------|-------------|
| `model` | estimator | Final model fitted on all data. |
| `metrics` | `Dict[str, float]` | Averaged metrics across CV folds. |
| `feature_columns` | `List[str]` | Final feature column names after encoding. |

**Example:**

```python
from modeling import fit_and_evaluate

result = fit_and_evaluate(
    X=fusion_df.drop(columns=["t", "unit_0"]),
    y=fusion_df["unit_0"].to_numpy(),
    model_name="xgboost",
    task="count",
    categorical_cols=["motif_id", "trial_type"],
)
print(f"CV R2: {result['metrics']['r2']:.3f}")
print(f"Features: {result['feature_columns'][:5]}...")
```

---

### `build_fusion_table`

```python
def build_fusion_table(
    spike_times: Dict[str, np.ndarray] | None,
    motifs: pd.DataFrame | None,
    bin_size_s: float,
) -> pd.DataFrame
```

Assemble the final neural-behavior fusion table by binning spike times and
behavioral motifs onto a common time grid.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spike_times` | `Dict[str, np.ndarray] \| None` | -- | Unit ID to spike times mapping. Used to determine the time range and compute spike counts. |
| `motifs` | `pd.DataFrame \| None` | -- | Behavioral motif table with a `t` column and feature columns (e.g., `motif_id`, `pose_speed`). |
| `bin_size_s` | `float` | -- | Time-bin width in seconds. |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | Fusion table with columns: `t` (bin center), one column per unit (spike counts), and all columns from `motifs` (binned by mean aggregation). |

**Example:**

```python
from modeling import build_fusion_table

fusion = build_fusion_table(
    spike_times=spikes,
    motifs=motif_df,
    bin_size_s=0.025,
)
print(f"Fusion table: {fusion.shape}")
print(fusion.columns.tolist()[:10])
# ['t', 'unit_0', 'unit_1', ..., 'motif_id', 'pose_speed']
```
