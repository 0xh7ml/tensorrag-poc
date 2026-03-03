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

class ForwardPassCard(BaseCard):
    card_type = "forward_pass"
    display_name = "Forward Pass"
    description = "Run data through the network"
    category = "training"
    execution_mode = "local"
    output_view_type = "metrics"

    config_schema = {}
    input_schema = {"training_state": "json"}
    output_schema = {"training_state": "json"}

    def execute(self, config, inputs, storage):
        state = storage.load_json(inputs["training_state"])
        arch = state["arch"]

        model = _make_model(arch)
        sd = {k: torch.tensor(v) for k, v in state["model_state_dict"].items()}
        model.load_state_dict(sd)
        model.train()

        X = torch.tensor(state["X"], dtype=torch.float32)
        logits = model(X)

        state["logits"] = logits.detach().tolist()
        ref = storage.save_json("_p", "_n", "training_state", state)
        return {"training_state": ref}

    def get_output_preview(self, outputs, storage):
        state = storage.load_json(outputs["training_state"])
        logits = state.get("logits", [])
        return {
            "batch_size": len(logits),
            "output_dim": len(logits[0]) if logits else 0,
            "status": "Forward pass complete",
        }
