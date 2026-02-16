# Card Development Guide

Build custom pipeline components for TensorRag. Each card is a single Python file that defines one processing step — load data, transform it, train a model, evaluate, or anything else you need.

## Quick Start

```python
from cards.base import BaseCard

class MyCard(BaseCard):
    card_type = "my_card"                # Unique identifier (snake_case)
    display_name = "My Card"             # Shown on canvas
    description = "What this card does"  # Shown in config panel
    category = "data"                    # Palette grouping
    execution_mode = "local"             # "local" or "modal"
    output_view_type = "table"           # How output is displayed

    config_schema = {}                   # User-configurable fields
    input_schema = {}                    # What this card receives
    output_schema = {}                   # What this card produces

    def execute(self, config, inputs, storage):
        # Your logic here
        return {}

    def get_output_preview(self, outputs, storage):
        # Return data for the Output tab
        return {}
```

Save this file in the Editor view, click **Validate**, then **Publish to Board**.

## Card Anatomy

### Required Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `card_type` | `str` | Unique snake_case identifier. This is how the system references your card. |
| `display_name` | `str` | Human-readable name shown on the canvas node. |
| `description` | `str` | Brief description shown in the Config panel. |
| `category` | `str` | One of: `data`, `model`, `training`, `evaluation`, `inference` |
| `execution_mode` | `str` | `"local"` (runs on backend server) or `"modal"` (runs on Modal serverless) |
| `output_view_type` | `str` | One of: `"table"`, `"metrics"`, `"model_summary"` |

### Required Methods

#### `execute(self, config, inputs, storage) -> dict`

The main logic. Receives configuration, input data references, and a storage object. Must return a dict mapping output names to storage references.

```python
def execute(self, config, inputs, storage):
    # config  — dict of user-configured values + _pipeline_id, _node_id
    # inputs  — dict mapping input names to storage references
    # storage — object with save/load methods

    df = storage.load_dataframe(inputs["dataset"])  # load input
    # ... do work ...
    ref = storage.save_dataframe("_p", "_n", "result", df)  # save output
    return {"result": ref}  # return output references
```

#### `get_output_preview(self, outputs, storage) -> dict`

Called after execution to generate a preview for the Output tab. The dict you return here is passed to the frontend viewer selected by `output_view_type`.

```python
def get_output_preview(self, outputs, storage):
    df = storage.load_dataframe(outputs["result"])
    return {
        "columns": list(df.columns),
        "rows": df.head(20).values.tolist(),
        "total_rows": len(df),
    }
```

## Schemas

### `config_schema` — User-Configurable Fields

Defines the form fields shown in the Config tab when a user clicks your card. Each key is a field name, and the value describes its type and default.

```python
config_schema = {
    "field_name": {
        "type": "string",           # "string", "number", or "boolean"
        "label": "Display Label",   # Shown in the form
        "default": "some value"     # Pre-filled value
    }
}
```
**Supported field types:**

| Type | Renders as | Python value |
|------|-----------|-------------|
| `"string"` | Text input | `str` |
| `"number"` | Number input | `float` or `int` |
| `"boolean"` | Checkbox | `bool` |

**Example:**

```python
config_schema = {
    "learning_rate": {
        "type": "number",
        "label": "Learning rate",
        "default": 0.01
    },
    "target_column": {
        "type": "string",
        "label": "Target column name",
        "default": "species"
    },
    "normalize": {
        "type": "boolean",
        "label": "Normalize features",
        "default": True
    }
}
```

No config? Use an empty dict:

```python
config_schema = {}
```

### `input_schema` — What Your Card Receives

Defines the input handles shown on the left side of your card node. Each key is the input name and the value is the data type.

```python
input_schema = {"dataset": "dataframe", "model": "model"}
```

**Supported data types:**

| Type | Description | Storage method |
|------|-------------|---------------|
| `"dataframe"` | Pandas DataFrame | `load_dataframe()` / `save_dataframe()` |
| `"model"` | Scikit-learn model or any picklable object | `load_model()` / `save_model()` |
| `"json"` | Dict / JSON-serializable data | `load_json()` / `save_json()` |

No inputs (source card)? Use an empty dict:

```python
input_schema = {}
```

### `output_schema` — What Your Card Produces

Defines the output handles shown on the right side of your card node. Same format as `input_schema`.

```python
output_schema = {"train_data": "dataframe", "test_data": "dataframe"}
```

The keys in `output_schema` **must match** the keys returned by `execute()`.

## Storage API

The `storage` object is passed to both `execute()` and `get_output_preview()`. Use it to save and load data between cards.

### Save Methods

All save methods return a **reference string** that you return from `execute()`.

```python
# Save a pandas DataFrame (stored as Parquet)
ref = storage.save_dataframe(pipeline_id, node_id, key, df)

# Save a model (stored via joblib — works with sklearn, PyTorch state_dicts, etc.)
ref = storage.save_model(pipeline_id, node_id, key, model)

# Save a JSON-serializable dict
ref = storage.save_json(pipeline_id, node_id, key, {"accuracy": 0.95})

# Save raw bytes (images, custom formats)
ref = storage.save_bytes(pipeline_id, node_id, key, png_bytes, "png")
```

> **Note:** For `pipeline_id` and `node_id`, use any placeholder like `"_p"` and `"_n"`. The executor automatically injects the real values at runtime.

### Load Methods

```python
df    = storage.load_dataframe(ref)   # Returns pandas DataFrame
model = storage.load_model(ref)       # Returns deserialized model
data  = storage.load_json(ref)        # Returns dict
raw   = storage.load_bytes(ref)       # Returns bytes
```

In `execute()`, load inputs using the references from `inputs`:

```python
df = storage.load_dataframe(inputs["dataset"])
```

In `get_output_preview()`, load outputs using the references from `outputs`:

```python
df = storage.load_dataframe(outputs["result"])
```

## Output View Types

The `output_view_type` attribute determines how your card's output is displayed in the Output tab. Your `get_output_preview()` return value must match the expected format.

### `"table"` — Tabular Data

Best for: datasets, DataFrames, any row/column data.

```python
output_view_type = "table"
```

**Return format (option A — column names + row arrays):**

```python
def get_output_preview(self, outputs, storage):
    df = storage.load_dataframe(outputs["dataset"])
    return {
        "columns": list(df.columns),                  # list of strings
        "rows": df.head(50).values.tolist(),           # list of lists
        "total_rows": len(df),                         # total count
    }
```

**Return format (option B — typed columns + dict rows):**

```python
def get_output_preview(self, outputs, storage):
    df = storage.load_dataframe(outputs["dataset"])
    return {
        "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
        "rows": df.head(50).to_dict(orient="records"),
        "shape": {"rows": len(df), "cols": len(df.columns)},
    }
```

**Return format (option C — train/test split):**

```python
def get_output_preview(self, outputs, storage):
    return {
        "train": {"rows": train_records, "row_count": 120},
        "test":  {"rows": test_records,  "row_count": 30},
        "split_ratio": {"train": 0.8, "test": 0.2},
    }
```

All three formats are auto-detected by the frontend.

### `"metrics"` — Key-Value Metrics

Best for: evaluation results, loss values, training statistics, any numeric results.

```python
output_view_type = "metrics"
```

**Return format (flat — recommended):**

Each key-value pair becomes a metric card in a 2-column grid.

```python
def get_output_preview(self, outputs, storage):
    return {
        "accuracy": 0.9667,
        "test_loss": 0.1234,
        "test_samples": 30,
        "status": "Evaluation complete",     # strings work too
    }
```

**Return format (wrapped):**

```python
def get_output_preview(self, outputs, storage):
    return {
        "metrics": {"accuracy": 0.9667, "f1_score": 0.95},
        "coefficients": {"feature_1": 0.45, "feature_2": -0.12},  # optional
        "intercept": 1.23,                                         # optional
        "chart_ref": "eval_chart",                                 # optional
    }
```

**Optional sections:**

| Key | Type | Renders as |
|-----|------|-----------|
| `metrics` | `dict[str, number]` | Grid of metric cards |
| `coefficients` | `dict[str, number]` | List of feature coefficients |
| `intercept` | `number` | Shown below coefficients |
| `gradient_norms` | `dict[str, number]` | Layer-wise gradient norms |
| `chart_ref` | `str` | Renders a chart image from artifacts |

### `"model_summary"` — Model Information

Best for: model architecture, hyperparameters, model metadata.

```python
output_view_type = "model_summary"
```

**Return format (wrapped):**

```python
def get_output_preview(self, outputs, storage):
    return {
        "model_type": "Neural Network",
        "hyperparameters": {
            "hidden_layers": "16, 8",
            "learning_rate": 0.01,
            "optimizer": "Adam",
        },
    }
```

**Return format (flat — also works):**

All top-level keys are treated as hyperparameters.

```python
def get_output_preview(self, outputs, storage):
    return {
        "architecture": "4 -> 16 -> 8 -> 3",
        "total_parameters": 155,
        "learning_rate": 0.01,
    }
```

## Categories

Categories control where your card appears in the sidebar palette.

| Category | Use for |
|----------|---------|
| `data` | Loading, cleaning, splitting, transforming datasets |
| `model` | Defining model architecture, building models |
| `training` | Forward pass, loss, backward pass, optimizer, training loops |
| `evaluation` | Testing, metrics, validation, benchmarks |
| `inference` | Prediction, serving, applying trained models |

## Execution Modes

| Mode | Where it runs | When to use |
|------|--------------|-------------|
| `"local"` | Backend server | Fast operations, small data, no GPU needed |
| `"modal"` | Modal serverless | Heavy computation, GPU needed, long-running tasks |

Cards running on Modal have access to: `pandas`, `numpy`, `scikit-learn`, `torch`, `torchvision`, `matplotlib`, `joblib`, `pyarrow`.

GPU cards use Modal's T4 GPU with CUDA support and have a 10-minute timeout.

## Wiring Cards Together

Cards connect through their input/output handles on the canvas:

```
[Card A] --output_name--> [Card B]
              ↑                ↑
     matches output_schema    matches input_schema
     key of Card A            key of Card B
```

**Rules:**
- An output handle can connect to multiple inputs (fan-out)
- An input handle receives from exactly one output
- Data types should match (e.g., `"dataframe"` output → `"dataframe"` input)
- The system validates all required inputs are connected before execution

**Example:**

```python
# Card A
output_schema = {"dataset": "dataframe"}

# Card B
input_schema = {"dataset": "dataframe"}
```

Drag from Card A's `dataset` output handle to Card B's `dataset` input handle on the canvas.

## Complete Examples

### Example 1: CSV Loader (Data Source)

A source card with no inputs — just configuration.

```python
from cards.base import BaseCard
import pandas as pd

class CsvLoaderCard(BaseCard):
    card_type = "csv_loader"
    display_name = "CSV Loader"
    description = "Load a dataset from a CSV URL"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "url": {
            "type": "string",
            "label": "CSV URL",
            "default": "https://example.com/data.csv"
        }
    }
    input_schema = {}
    output_schema = {"dataset": "dataframe"}

    def execute(self, config, inputs, storage):
        df = pd.read_csv(config["url"])
        ref = storage.save_dataframe("_p", "_n", "dataset", df)
        return {"dataset": ref}

    def get_output_preview(self, outputs, storage):
        df = storage.load_dataframe(outputs["dataset"])
        return {
            "columns": list(df.columns),
            "rows": df.head(20).values.tolist(),
            "total_rows": len(df),
        }
```

### Example 2: Feature Scaler (Transform)

Takes a DataFrame, processes it, outputs a DataFrame.

```python
from cards.base import BaseCard
import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeatureScalerCard(BaseCard):
    card_type = "feature_scaler"
    display_name = "Feature Scaler"
    description = "Standardize numeric features (zero mean, unit variance)"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "columns": {
            "type": "string",
            "label": "Columns to scale (comma-separated, empty = all numeric)",
            "default": ""
        }
    }
    input_schema = {"dataset": "dataframe"}
    output_schema = {"scaled_dataset": "dataframe"}

    def execute(self, config, inputs, storage):
        df = storage.load_dataframe(inputs["dataset"])
        cols_str = config.get("columns", "").strip()

        if cols_str:
            columns = [c.strip() for c in cols_str.split(",")]
        else:
            columns = df.select_dtypes(include="number").columns.tolist()

        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])

        ref = storage.save_dataframe("_p", "_n", "scaled_dataset", df)
        return {"scaled_dataset": ref}

    def get_output_preview(self, outputs, storage):
        df = storage.load_dataframe(outputs["scaled_dataset"])
        return {
            "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
            "rows": df.head(50).to_dict(orient="records"),
            "shape": {"rows": len(df), "cols": len(df.columns)},
        }
```

### Example 3: Model Trainer (Multiple Inputs/Outputs)

Receives data, trains a model, outputs both the model and metrics.

```python
from cards.base import BaseCard
from sklearn.ensemble import RandomForestClassifier

class RandomForestCard(BaseCard):
    card_type = "random_forest"
    display_name = "Random Forest"
    description = "Train a random forest classifier"
    category = "training"
    execution_mode = "modal"
    output_view_type = "metrics"

    config_schema = {
        "n_estimators": {
            "type": "number",
            "label": "Number of trees",
            "default": 100
        },
        "max_depth": {
            "type": "number",
            "label": "Max tree depth (0 = unlimited)",
            "default": 0
        }
    }
    input_schema = {"train_data": "json"}
    output_schema = {"trained_model": "model", "train_metrics": "json"}

    def execute(self, config, inputs, storage):
        data = storage.load_json(inputs["train_data"])

        n_est = int(config.get("n_estimators", 100))
        max_d = int(config.get("max_depth", 0)) or None

        model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d)
        model.fit(data["X"], data["y"])

        train_acc = model.score(data["X"], data["y"])
        metrics = {
            "train_accuracy": round(train_acc, 4),
            "n_estimators": n_est,
            "n_features": len(data["X"][0]),
        }

        model_ref = storage.save_model("_p", "_n", "trained_model", model)
        metrics_ref = storage.save_json("_p", "_n", "train_metrics", metrics)
        return {"trained_model": model_ref, "train_metrics": metrics_ref}

    def get_output_preview(self, outputs, storage):
        return storage.load_json(outputs["train_metrics"])
```

### Example 4: Evaluator with Chart

Generates a confusion matrix image alongside metrics.

```python
from cards.base import BaseCard
import numpy as np

class ConfusionMatrixCard(BaseCard):
    card_type = "confusion_matrix"
    display_name = "Confusion Matrix"
    description = "Evaluate with confusion matrix visualization"
    category = "evaluation"
    execution_mode = "modal"
    output_view_type = "metrics"

    config_schema = {}
    input_schema = {"trained_model": "model", "test_data": "json"}
    output_schema = {"metrics": "json"}

    def execute(self, config, inputs, storage):
        from sklearn.metrics import accuracy_score, confusion_matrix
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io

        model = storage.load_model(inputs["trained_model"])
        test = storage.load_json(inputs["test_data"])

        preds = model.predict(test["X"])
        y_true = test["y"]
        acc = accuracy_score(y_true, preds)
        cm = confusion_matrix(y_true, preds)

        # Generate chart
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        storage.save_bytes("_p", "_n", "eval_chart", buf.getvalue(), "png")

        metrics = {"accuracy": round(acc, 4), "classes": int(cm.shape[0])}
        ref = storage.save_json("_p", "_n", "metrics", metrics)
        return {"metrics": ref}

    def get_output_preview(self, outputs, storage):
        m = storage.load_json(outputs["metrics"])
        return {
            "metrics": m,
            "chart_ref": "eval_chart",   # Renders the saved PNG in the Output tab
        }
```

### Example 5: JSON Pass-Through (Minimal)

Simplest possible card — transforms JSON data.

```python
from cards.base import BaseCard

class FilterColumnsCard(BaseCard):
    card_type = "filter_columns"
    display_name = "Filter Columns"
    description = "Keep only specified columns"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "keep_columns": {
            "type": "string",
            "label": "Columns to keep (comma-separated)",
            "default": ""
        }
    }
    input_schema = {"dataset": "dataframe"}
    output_schema = {"dataset": "dataframe"}

    def execute(self, config, inputs, storage):
        import pandas as pd
        df = storage.load_dataframe(inputs["dataset"])
        cols = [c.strip() for c in config["keep_columns"].split(",") if c.strip()]
        if cols:
            df = df[cols]
        ref = storage.save_dataframe("_p", "_n", "dataset", df)
        return {"dataset": ref}

    def get_output_preview(self, outputs, storage):
        import pandas as pd
        df = storage.load_dataframe(outputs["dataset"])
        return {
            "columns": list(df.columns),
            "rows": df.head(20).values.tolist(),
            "total_rows": len(df),
        }
```

---

## Tips

- **Class name doesn't matter.** Only `card_type` is used for identification. Name your class whatever you want.
- **One card per file.** The validator rejects files with multiple `BaseCard` subclasses.
- **Use `"_p"` and `"_n"` as placeholders** for `pipeline_id` and `node_id` in storage calls. The executor replaces them automatically.
- **Imports go at the top** or inside methods. For Modal execution, stick to packages in the Modal image (pandas, numpy, scikit-learn, torch, matplotlib, joblib).
- **Helper functions are fine.** Define utility functions outside the class in the same file (e.g., `_make_model(arch)`).
- **String columns from Parquet** come back as `string` dtype, not `object`. Use `pd.api.types.is_numeric_dtype(col)` instead of `dtype == object` for type checks.
- **Config values are strings by default.** Always cast: `float(config.get("lr", 0.01))`, `int(config.get("epochs", 50))`.
- **Large data on Modal.** Data is serialized and sent to Modal containers. Keep intermediate data reasonable — don't pass multi-GB DataFrames through JSON.
- **Charts.** Save PNG bytes with `storage.save_bytes()` and reference them in preview with `"chart_ref": "key_name"`. The Output tab renders them automatically.

## Validation Checklist

When you click **Validate** in the editor, the system checks:

- [ ] Valid Python syntax
- [ ] Exactly one class extending `BaseCard`
- [ ] All 6 required attributes present (`card_type`, `display_name`, `description`, `category`, `execution_mode`, `output_view_type`)
- [ ] `category` is one of: `data`, `model`, `training`, `evaluation`, `inference`
- [ ] `execution_mode` is one of: `local`, `modal`
- [ ] `output_view_type` is one of: `table`, `metrics`, `model_summary`
- [ ] Both `execute()` and `get_output_preview()` methods exist
- [ ] Schema attributes (`config_schema`, `input_schema`, `output_schema`) are dict literals
