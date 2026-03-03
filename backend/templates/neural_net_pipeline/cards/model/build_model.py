from cards.base import BaseCard
import torch
import torch.nn as nn
import numpy as np

def _make_model(arch):
    """Build a Sequential model from arch list, e.g. [4, 16, 3]."""
    layers = []
    for i in range(len(arch) - 1):
        layers.append(nn.Linear(arch[i], arch[i + 1]))
        if i < len(arch) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class BuildModelCard(BaseCard):
    card_type = "build_model"
    display_name = "Build Model"
    description = "Create a neural network architecture"
    category = "model"
    execution_mode = "local"
    output_view_type = "model_summary"

    config_schema = {
        "hidden_sizes": {
            "type": "string",
            "label": "Hidden layer sizes (comma-separated)",
            "default": "16,8"
        },
        "learning_rate": {
            "type": "number",
            "label": "Learning rate",
            "default": 0.01
        }
    }
    input_schema = {"train_data": "json"}
    output_schema = {"training_state": "json"}

    def execute(self, config, inputs, storage):
        train = storage.load_json(inputs["train_data"])
        n_features = train["num_features"]
        n_classes  = train["num_classes"]
        hidden = [int(h.strip()) for h in config["hidden_sizes"].split(",")]
        lr = float(config.get("learning_rate", 0.01))

        arch = [n_features] + hidden + [n_classes]
        model = _make_model(arch)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        state = {
            "model_state_dict": {k: v.tolist() for k, v in model.state_dict().items()},
            "optimizer_state_dict": None,
            "arch": arch,
            "lr": lr,
            "X": train["X"],
            "y": train["y"],
        }
        ref = storage.save_json("_p", "_n", "training_state", state)
        return {"training_state": ref}

    def get_output_preview(self, outputs, storage):
        state = storage.load_json(outputs["training_state"])
        arch = state["arch"]
        total = sum(arch[i] * arch[i+1] + arch[i+1] for i in range(len(arch)-1))
        return {
            "architecture": " -> ".join(str(a) for a in arch),
            "total_parameters": total,
            "learning_rate": state["lr"],
        }
