from cards.base import BaseCard
import pandas as pd
import numpy as np

class DataSplitCard(BaseCard):
    card_type = "data_split"
    display_name = "Data Split"
    description = "Split dataset into train and test sets"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "target_column": {
            "type": "string",
            "label": "Target column name",
            "default": "species"
        },
        "test_ratio": {
            "type": "number",
            "label": "Test set ratio",
            "default": 0.2
        }
    }
    input_schema = {"dataset": "dataframe"}
    output_schema = {"train_data": "json", "test_data": "json"}

    def execute(self, config, inputs, storage):
        df = storage.load_dataframe(inputs["dataset"])
        target = config["target_column"]
        ratio = float(config.get("test_ratio", 0.2))

        if df[target].dtype == object:
            labels = sorted(df[target].unique().tolist())
            label_map = {l: i for i, l in enumerate(labels)}
            df[target] = df[target].map(label_map)

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split = int(len(df) * (1 - ratio))

        X = df.drop(columns=[target]).values.astype(float)
        y = df[target].values.astype(float)

        train = {"X": X[:split].tolist(), "y": y[:split].tolist(),
                 "feature_names": [c for c in df.columns if c != target],
                 "num_features": X.shape[1],
                 "num_classes": int(y.max()) + 1}
        test  = {"X": X[split:].tolist(), "y": y[split:].tolist(),
                 "feature_names": train["feature_names"],
                 "num_features": train["num_features"],
                 "num_classes": train["num_classes"]}

        ref_train = storage.save_json("_p", "_n", "train_data", train)
        ref_test  = storage.save_json("_p", "_n", "test_data", test)
        return {"train_data": ref_train, "test_data": ref_test}

    def get_output_preview(self, outputs, storage):
        train = storage.load_json(outputs["train_data"])
        test  = storage.load_json(outputs["test_data"])
        return {
            "columns": ["split", "samples", "features", "classes"],
            "rows": [
                ["Train", len(train["y"]), train["num_features"], train["num_classes"]],
                ["Test",  len(test["y"]),  test["num_features"],  test["num_classes"]],
            ],
            "total_rows": 2,
        }
