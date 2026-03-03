from cards.base import BaseCard
import pandas as pd

class DataLoadCard(BaseCard):
    card_type = "data_load"
    display_name = "Data Load"
    description = "Load a CSV dataset"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "source_url": {
            "type": "string",
            "label": "CSV URL or path",
            "default": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        }
    }
    input_schema = {}
    output_schema = {"dataset": "dataframe"}

    def execute(self, config, inputs, storage):
        df = pd.read_csv(config["source_url"])
        ref = storage.save_dataframe("_p", "_n", "dataset", df)
        return {"dataset": ref}

    def get_output_preview(self, outputs, storage):
        df = storage.load_dataframe(outputs["dataset"])
        return {
            "columns": list(df.columns),
            "rows": df.head(20).values.tolist(),
            "total_rows": len(df),
        }
