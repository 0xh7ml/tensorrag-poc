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

class EvaluateCard(BaseCard):
    card_type = "evaluate"
    display_name = "Evaluate"
    description = "Test model on held-out data"
    category = "evaluation"
    execution_mode = "local"
    output_view_type = "metrics"

    config_schema = {}
    input_schema = {"trained_model": "json", "test_data": "json"}
    output_schema = {"metrics": "json"}

    def execute(self, config, inputs, storage):
        trained = storage.load_json(inputs["trained_model"])
        test = storage.load_json(inputs["test_data"])

        model = _make_model(trained["arch"])
        sd = {k: torch.tensor(v) for k, v in trained["model_state_dict"].items()}
        model.load_state_dict(sd)
        model.eval()

        X = torch.tensor(test["X"], dtype=torch.float32)
        y_true = np.array(test["y"], dtype=int)

        with torch.no_grad():
            logits = model(X)
            loss = nn.CrossEntropyLoss()(logits, torch.tensor(test["y"], dtype=torch.long)).item()
            preds = logits.argmax(dim=1).numpy()

        accuracy = float((preds == y_true).mean())
        metrics = {
            "accuracy": round(accuracy, 4),
            "test_loss": round(loss, 6),
            "test_samples": len(y_true),
        }
        ref = storage.save_json("_p", "_n", "metrics", metrics)
        return {"metrics": ref}

    def get_output_preview(self, outputs, storage):
        return storage.load_json(outputs["metrics"])
