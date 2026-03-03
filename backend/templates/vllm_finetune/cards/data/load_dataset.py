from cards.base import BaseCard

class LoadDatasetCard(BaseCard):
    card_type = "llm_load_dataset"
    display_name = "Load Dataset"
    description = "Load dataset from HuggingFace or file"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "source": {
            "type": "string",
            "label": "HuggingFace dataset name or file URL",
            "default": "tatsu-lab/alpaca"
        },
        "split": {
            "type": "string",
            "label": "Split",
            "default": "train"
        },
        "max_samples": {
            "type": "number",
            "label": "Max samples (0 = all)",
            "default": 1000
        }
    }
    input_schema = {}
    output_schema = {"dataset": "json"}

    def execute(self, config, inputs, storage):
        from datasets import load_dataset

        source = config["source"]
        split = config.get("split", "train")
        max_samples = int(config.get("max_samples", 1000))

        ds = load_dataset(source, split=split)

        if max_samples > 0:
            ds = ds.select(range(min(max_samples, len(ds))))

        records = [dict(row) for row in ds]

        data = {
            "records": records,
            "num_samples": len(records),
            "columns": list(records[0].keys()) if records else [],
            "source": source,
        }
        ref = storage.save_json("_p", "_n", "dataset", data)
        return {"dataset": ref}

    def get_output_preview(self, outputs, storage):
        data = storage.load_json(outputs["dataset"])
        records = data["records"][:10]
        columns = data["columns"]

        rows = []
        for r in records:
            row = []
            for c in columns:
                val = str(r.get(c, ""))
                row.append(val[:100] + "..." if len(val) > 100 else val)
            rows.append(row)

        return {
            "columns": columns,
            "rows": rows,
            "total_rows": data["num_samples"],
        }
