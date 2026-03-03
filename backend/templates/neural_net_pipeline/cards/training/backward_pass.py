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

class BackwardPassCard(BaseCard):
    card_type = "backward_pass"
    display_name = "Backward Pass"
    description = "Compute gradients via backpropagation"
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
        y = torch.tensor(state["y"], dtype=torch.long)

        logits = model(X)
        loss = nn.CrossEntropyLoss()(logits, y)

        loss.backward()

        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.tolist()

        state["gradients"] = grads
        state["loss"] = loss.item()
        state["model_state_dict"] = {k: v.tolist() for k, v in model.state_dict().items()}

        ref = storage.save_json("_p", "_n", "training_state", state)
        return {"training_state": ref}

    def get_output_preview(self, outputs, storage):
        state = storage.load_json(outputs["training_state"])
        grads = state.get("gradients", {})
        grad_norms = {}
        for name, g in grads.items():
            t = torch.tensor(g)
            grad_norms[name] = round(t.norm().item(), 6)
        return {
            "loss": round(state.get("loss", 0), 6),
            "gradient_norms": grad_norms,
            "status": "Gradients computed",
        }
