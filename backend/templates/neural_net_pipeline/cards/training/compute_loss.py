from cards.base import BaseCard
import torch
import torch.nn as nn
import numpy as np

class ComputeLossCard(BaseCard):
    card_type = "compute_loss"
    display_name = "Compute Loss"
    description = "Calculate cross-entropy loss"
    category = "training"
    execution_mode = "local"
    output_view_type = "metrics"

    config_schema = {}
    input_schema = {"training_state": "json"}
    output_schema = {"training_state": "json"}

    def execute(self, config, inputs, storage):
        state = storage.load_json(inputs["training_state"])

        logits = torch.tensor(state["logits"], dtype=torch.float32)
        y = torch.tensor(state["y"], dtype=torch.long)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, y)

        state["loss"] = loss.item()
        ref = storage.save_json("_p", "_n", "training_state", state)
        return {"training_state": ref}

    def get_output_preview(self, outputs, storage):
        state = storage.load_json(outputs["training_state"])
        return {
            "loss": round(state.get("loss", 0), 6),
            "status": "Loss computed",
        }
