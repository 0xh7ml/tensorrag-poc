from cards.base import BaseCard
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

class InferenceCard(BaseCard):
    card_type = "cv_inference"
    display_name = "Image Inference"
    description = "Classify new images"
    category = "inference"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "image_url": {
            "type": "string",
            "label": "Image URL",
            "default": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
        }
    }
    input_schema = {"trained_model": "json"}
    output_schema = {"predictions": "json"}

    def execute(self, config, inputs, storage):
        model_info = storage.load_json(inputs["trained_model"])
        image_url = config.get("image_url")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        arch = model_info["architecture"]
        num_classes = model_info["num_classes"]

        if arch == "resnet18":
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif arch == "resnet50":
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif arch == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=False)
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

        state_dict = {k: torch.tensor(v) for k, v in model_info["trained_state"].items()}
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        if image_url.startswith("http"):
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_url).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()

        class_names = model_info["classes"]
        predictions = {
            "image_url": image_url,
            "predicted_class": predicted_class,
            "predicted_label": class_names[predicted_class],
            "confidence": float(probabilities[0][predicted_class]),
            "all_probabilities": {
                class_names[i]: float(prob)
                for i, prob in enumerate(probabilities[0])
            }
        }

        ref = storage.save_json("_p", "_n", "predictions", predictions)
        return {"predictions": ref}

    def get_output_preview(self, outputs, storage):
        pred = storage.load_json(outputs["predictions"])

        probs = pred["all_probabilities"]
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]

        rows = []
        for i, (class_name, prob) in enumerate(sorted_probs):
            rows.append([i+1, class_name, f"{prob:.4f}"])

        return {
            "columns": ["Rank", "Class", "Probability"],
            "rows": rows,
            "total_rows": len(rows),
            "predicted_class": pred["predicted_label"],
            "confidence": f"{pred['confidence']:.2%}"
        }
