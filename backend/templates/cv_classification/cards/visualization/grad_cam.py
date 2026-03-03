from cards.base import BaseCard
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import requests
from io import BytesIO
import base64

class GradCAMCard(BaseCard):
    card_type = "cv_grad_cam"
    display_name = "Grad-CAM Visualization"
    description = "Generate attention heatmaps"
    category = "evaluation"
    execution_mode = "local"
    output_view_type = "metrics"

    config_schema = {
        "image_url": {
            "type": "string",
            "label": "Image URL for visualization",
            "default": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
        },
        "target_layer": {
            "type": "string",
            "label": "Target layer name",
            "default": "layer4"
        }
    }
    input_schema = {"trained_model": "json"}
    output_schema = {"heatmaps": "json"}

    def execute(self, config, inputs, storage):
        model_info = storage.load_json(inputs["trained_model"])
        image_url = config.get("image_url")
        target_layer_name = config.get("target_layer", "layer4")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        arch = model_info["architecture"]
        num_classes = model_info["num_classes"]

        if arch == "resnet18":
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            target_layer = model.layer4[-1].conv2
        elif arch == "resnet50":
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            target_layer = model.layer4[-1].conv3
        else:
            raise ValueError(f"Grad-CAM not implemented for {arch}")

        state_dict = {k: torch.tensor(v) for k, v in model_info["trained_state"].items()}
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        if image_url.startswith("http"):
            response = requests.get(image_url)
            original_image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            original_image = Image.open(image_url).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_tensor = transform(original_image).unsqueeze(0).to(device)
        input_tensor.requires_grad_()

        gradients = []
        activations = []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        def forward_hook(module, input, output):
            activations.append(output)

        handle_backward = target_layer.register_backward_hook(backward_hook)
        handle_forward = target_layer.register_forward_hook(forward_hook)

        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

        model.zero_grad()
        output[0, predicted_class].backward()

        grad = gradients[0].squeeze()
        activation = activations[0].squeeze()

        weights = torch.mean(grad, dim=[1, 2])
        cam = torch.sum(weights.unsqueeze(1).unsqueeze(2) * activation, dim=0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam

        cam_resized = nn.functional.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode='bilinear'
        ).squeeze().cpu().numpy()

        original_resized = np.array(original_image.resize((224, 224)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = heatmap * 0.4 + original_resized * 0.6
        overlay = np.uint8(overlay)

        def numpy_to_base64(img_array):
            img = Image.fromarray(img_array)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()

        results = {
            "image_url": image_url,
            "predicted_class": predicted_class,
            "predicted_label": model_info["classes"][predicted_class],
            "original_image_b64": numpy_to_base64(original_resized),
            "heatmap_b64": numpy_to_base64(heatmap),
            "overlay_b64": numpy_to_base64(overlay),
            "target_layer": target_layer_name
        }

        handle_backward.remove()
        handle_forward.remove()

        ref = storage.save_json("_p", "_n", "heatmaps", results)
        return {"heatmaps": ref}

    def get_output_preview(self, outputs, storage):
        data = storage.load_json(outputs["heatmaps"])
        return {
            "predicted_class": data["predicted_label"],
            "target_layer": data["target_layer"],
            "visualization_generated": "Yes",
            "image_source": data["image_url"][:50] + "..." if len(data["image_url"]) > 50 else data["image_url"]
        }
