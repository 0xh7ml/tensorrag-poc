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

class OptimizerStepCard(BaseCard):
    card_type = "optimizer_step"
    display_name = "Optimizer Step"
    description = "Update weights using gradients (multi-epoch)"
    category = "training"
    execution_mode = "local"
    output_view_type = "metrics"

    config_schema = {
        "epochs": {
            "type": "number",
            "label": "Number of epochs",
            "default": 50
        }
    }
    input_schema = {"training_state": "json"}
    output_schema = {"trained_model": "json"}

    def execute(self, config, inputs, storage):
        state = storage.load_json(inputs["training_state"])
        arch = state["arch"]
        lr = state["lr"]
        epochs = int(config.get("epochs", 50))

        model = _make_model(arch)
        sd = {k: torch.tensor(v) for k, v in state["model_state_dict"].items()}
        model.load_state_dict(sd)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        X = torch.tensor(state["X"], dtype=torch.float32)
        y = torch.tensor(state["y"], dtype=torch.long)

        history = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            history.append(round(loss.item(), 6))

        trained = {
            "model_state_dict": {k: v.tolist() for k, v in model.state_dict().items()},
            "arch": arch,
            "lr": lr,
            "loss_history": history,
            "final_loss": history[-1],
        }
        ref = storage.save_json("_p", "_n", "trained_model", trained)
        return {"trained_model": ref}

    def get_output_preview(self, outputs, storage):
        data = storage.load_json(outputs["trained_model"])
        h = data.get("loss_history", [])
        return {
            "epochs": len(h),
            "initial_loss": h[0] if h else None,
            "final_loss": h[-1] if h else None,
            "status": "Training complete",
        }
