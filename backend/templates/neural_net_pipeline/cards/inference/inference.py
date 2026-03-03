from cards.base import BaseCard
import torch
import torch.nn as nn
import numpy as np

def _make_model(arch):
    layers = []
    for i in range(len(arch) - 1):
        layers.append(nn.Linear(arch[i], arch[i + 1]))
        if i < len(arch) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class InferenceCard(BaseCard):
    card_type = "inference"
    display_name = "Inference"
    description = "Make predictions with trained model"
    category = "inference"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "input_values": {
            "type": "string",
            "label": "Input values (comma-separated)",
            "default": "5.1,3.5,1.4,0.2"
        }
    }
    input_schema = {"trained_model": "json"}
    output_schema = {"predictions": "json"}

    def execute(self, config, inputs, storage):
        trained = storage.load_json(inputs["trained_model"])

        model = _make_model(trained["arch"])
        sd = {k: torch.tensor(v) for k, v in trained["model_state_dict"].items()}
        model.load_state_dict(sd)
        model.eval()

        raw = config.get("input_values", "")
        values = [float(v.strip()) for v in raw.split(",")]
        X = torch.tensor([values], dtype=torch.float32)

        with torch.no_grad():
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()

        result = {
            "input": values,
            "predicted_class": pred_class,
            "probabilities": probs[0].tolist(),
        }
        ref = storage.save_json("_p", "_n", "predictions", result)
        return {"predictions": ref}

    def get_output_preview(self, outputs, storage):
        result = storage.load_json(outputs["predictions"])
        probs = result.get("probabilities", [])
        rows = [[i, round(p, 4)] for i, p in enumerate(probs)]
        return {
            "columns": ["class", "probability"],
            "rows": rows,
            "total_rows": len(rows),
            "predicted_class": result.get("predicted_class"),
        }
