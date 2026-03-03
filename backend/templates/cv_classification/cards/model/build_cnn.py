from cards.base import BaseCard
import torch
import torch.nn as nn
import torchvision.models as models

class BuildCNNCard(BaseCard):
    card_type = "cv_build_cnn"
    display_name = "Build CNN Model"
    description = "Create CNN architecture"
    category = "model"
    execution_mode = "local"
    output_view_type = "model_summary"

    config_schema = {
        "architecture": {
            "type": "string",
            "label": "CNN Architecture",
            "default": "resnet18"
        },
        "pretrained": {
            "type": "boolean",
            "label": "Use pretrained weights",
            "default": True
        },
        "learning_rate": {
            "type": "number",
            "label": "Learning rate",
            "default": 0.001
        }
    }
    input_schema = {"augmented_data": "json"}
    output_schema = {"model_config": "json"}

    def execute(self, config, inputs, storage):
        data_info = storage.load_json(inputs["augmented_data"])
        arch = config.get("architecture", "resnet18")
        pretrained = config.get("pretrained", True)
        lr = float(config.get("learning_rate", 0.001))

        num_classes = data_info["num_classes"]

        if arch == "resnet18":
            model = models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif arch == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif arch == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            class SimpleCNN(nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1)
                    )
                    self.classifier = nn.Linear(128, num_classes)

                def forward(self, x):
                    x = self.features(x)
                    x = x.view(x.size(0), -1)
                    x = self.classifier(x)
                    return x

            model = SimpleCNN(num_classes)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model_info = {
            **data_info,
            "architecture": arch,
            "pretrained": pretrained,
            "learning_rate": lr,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_state": {k: v.tolist() for k, v in model.state_dict().items()}
        }

        ref = storage.save_json("_p", "_n", "model_config", model_info)
        return {"model_config": ref}

    def get_output_preview(self, outputs, storage):
        info = storage.load_json(outputs["model_config"])
        return {
            "architecture": info["architecture"],
            "pretrained": "Yes" if info["pretrained"] else "No",
            "total_parameters": f"{info['total_parameters']:,}",
            "trainable_parameters": f"{info['trainable_parameters']:,}",
            "learning_rate": info["learning_rate"],
            "num_classes": info["num_classes"]
        }
