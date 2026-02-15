# How to Create a Custom Card

A card is a single Python file with one class that extends `BaseCard`. That's it.

## Minimal Template

```python
from cards.base import BaseCard

class MyCard(BaseCard):
    # --- Required attributes ---
    card_type = "my_card"              # Unique ID (snake_case)
    display_name = "My Card"           # Shown in the UI
    description = "What this card does"
    category = "data"                  # data | model | training | evaluation | inference
    execution_mode = "local"           # local | modal
    output_view_type = "table"         # table | metrics | model_summary

    # --- Schemas (what the card accepts and produces) ---
    config_schema = {}    # User-configurable settings
    input_schema = {}     # What this card receives from other cards
    output_schema = {}    # What this card sends to other cards

    # --- Required methods ---
    def execute(self, config, inputs, storage):
        # Your logic here
        return {}

    def get_output_preview(self, outputs, storage):
        # Return data for the Output tab
        return {}
```

## Step by Step

### 1. Set the attributes

| Attribute | What it is | Example |
|-----------|-----------|---------|
| `card_type` | Unique identifier | `"csv_loader"` |
| `display_name` | UI label | `"CSV Loader"` |
| `description` | Short explanation | `"Loads data from CSV"` |
| `category` | Palette group | `"data"` |
| `execution_mode` | Where it runs | `"local"` |
| `output_view_type` | How output renders | `"table"` |

**Valid values:**
- `category`: `data`, `model`, `training`, `evaluation`, `inference`
- `execution_mode`: `local`, `modal`
- `output_view_type`: `table`, `metrics`, `model_summary`

### 2. Define schemas

Schemas are dicts that describe config fields, inputs, and outputs.

**config_schema** — user-editable fields shown in the sidebar:
```python
config_schema = {
    "file_url": {
        "type": "string",
        "label": "File URL",
        "default": "https://example.com/data.csv"
    },
    "batch_size": {
        "type": "number",
        "label": "Batch Size",
        "default": 32
    }
}
```

**input_schema** — what this card receives from connected cards:
```python
input_schema = {
    "dataset": "dataframe",
    "model": "json"
}
```

**output_schema** — what this card sends to downstream cards:
```python
output_schema = {
    "result": "dataframe",
    "metrics": "json"
}
```

### 3. Write `execute()`

This is where your logic runs. It receives:
- `config` — dict of user settings from `config_schema`
- `inputs` — dict of S3 references from upstream cards
- `storage` — the storage service for saving/loading data

**Returns:** a dict mapping output names to S3 references.

```python
def execute(self, config, inputs, storage):
    # Load input from upstream card
    df = storage.load_dataframe(inputs["dataset"])

    # Do something
    df = df.dropna()

    # Save output to S3
    ref = storage.save_dataframe("_p", "_n", "clean_data", df)
    return {"clean_data": ref}
```

### 4. Write `get_output_preview()`

This generates the data shown in the Output tab when a user clicks the card.

- `outputs` — same dict your `execute()` returned
- `storage` — use it to load the data back

**For `output_view_type = "table"`:**
```python
def get_output_preview(self, outputs, storage):
    df = storage.load_dataframe(outputs["clean_data"])
    return {
        "columns": list(df.columns),
        "rows": df.head(20).values.tolist(),
        "total_rows": len(df),
    }
```

**For `output_view_type = "metrics"`:**
```python
def get_output_preview(self, outputs, storage):
    metrics = storage.load_json(outputs["metrics"])
    return metrics  # e.g. {"accuracy": 0.95, "loss": 0.12}
```

**For `output_view_type = "model_summary"`:**
```python
def get_output_preview(self, outputs, storage):
    return {
        "architecture": "4 -> 16 -> 3",
        "total_parameters": 131,
        "learning_rate": 0.01,
    }
```

---

## Storage API Reference

The `storage` object provides these methods:

| Method | Save | Load |
|--------|------|------|
| **DataFrame** (Parquet) | `storage.save_dataframe(pid, nid, key, df)` | `storage.load_dataframe(ref)` |
| **JSON** | `storage.save_json(pid, nid, key, data)` | `storage.load_json(ref)` |
| **Model** (Joblib) | `storage.save_model(pid, nid, key, model)` | `storage.load_model(ref)` |
| **Binary** (PNG, etc.) | `storage.save_bytes(pid, nid, key, data, ext)` | `storage.load_bytes(ref)` |

All `save_*` methods return a string reference (S3 URI). Pass that reference to `load_*` to retrieve the data.

> For `pid` and `nid` parameters, use any placeholder string (e.g. `"_p"`, `"_n"`). The executor fills in the real pipeline/node IDs automatically.

---

## Complete Examples

### Example 1: Text Preprocessor

```python
from cards.base import BaseCard
import pandas as pd

class TextPreprocessorCard(BaseCard):
    card_type = "text_preprocessor"
    display_name = "Text Preprocessor"
    description = "Clean and normalize text data"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "text_column": {
            "type": "string",
            "label": "Text column name",
            "default": "text"
        },
        "lowercase": {
            "type": "boolean",
            "label": "Convert to lowercase",
            "default": True
        }
    }
    input_schema = {"dataset": "dataframe"}
    output_schema = {"dataset": "dataframe"}

    def execute(self, config, inputs, storage):
        df = storage.load_dataframe(inputs["dataset"])
        col = config["text_column"]

        if config.get("lowercase", True):
            df[col] = df[col].str.lower()

        df[col] = df[col].str.strip()

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

### Example 2: Accuracy Calculator

```python
from cards.base import BaseCard
import numpy as np

class AccuracyCard(BaseCard):
    card_type = "accuracy_calc"
    display_name = "Accuracy"
    description = "Calculate classification accuracy"
    category = "evaluation"
    execution_mode = "local"
    output_view_type = "metrics"

    config_schema = {}
    input_schema = {"predictions": "json", "ground_truth": "json"}
    output_schema = {"metrics": "json"}

    def execute(self, config, inputs, storage):
        preds = storage.load_json(inputs["predictions"])
        truth = storage.load_json(inputs["ground_truth"])

        y_pred = np.array(preds["values"])
        y_true = np.array(truth["values"])

        accuracy = float((y_pred == y_true).mean())
        metrics = {"accuracy": round(accuracy, 4), "samples": len(y_true)}

        ref = storage.save_json("_p", "_n", "metrics", metrics)
        return {"metrics": ref}

    def get_output_preview(self, outputs, storage):
        return storage.load_json(outputs["metrics"])
```

---

## Workflow

1. Open the **Editor** view (toggle in the header)
2. Click **New Card** — enter a name
3. Write your Python code following the template above
4. Click **Validate** — fix any errors shown
5. Click **Publish to Board**
6. Switch to **Board** view — your card appears in the palette
7. Drag it onto the canvas, connect it, and run

## Rules

- One class per file, extending `BaseCard`
- All 6 attributes are required
- Both `execute()` and `get_output_preview()` are required
- `config_schema`, `input_schema`, `output_schema` must be dict literals
- `card_type` must be unique across all cards
- Use `storage` for all data persistence — don't write to local disk
