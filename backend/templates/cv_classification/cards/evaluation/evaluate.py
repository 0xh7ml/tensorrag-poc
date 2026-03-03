from cards.base import BaseCard
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class EvaluateCard(BaseCard):
    card_type = "cv_evaluate"
    display_name = "Evaluate Model"
    description = "Evaluate on test dataset"
    category = "evaluation"
    execution_mode = "local"
    output_view_type = "metrics"

    config_schema = {}
    input_schema = {"trained_model": "json", "preprocessed_data": "json"}
    output_schema = {"metrics": "json"}

    def execute(self, config, inputs, storage):
        model_info = storage.load_json(inputs["trained_model"])
        data_info = storage.load_json(inputs["preprocessed_data"])

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

        resize_dim = data_info["resize_dim"]
        test_transform = transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        data_root = data_info["data_root"]
        dataset_type = data_info["dataset_type"]

        if dataset_type == "CIFAR10":
            test_dataset = datasets.CIFAR10(data_root, train=False, transform=test_transform)
        elif dataset_type == "CIFAR100":
            test_dataset = datasets.CIFAR100(data_root, train=False, transform=test_transform)

        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()

                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        test_loss /= len(test_loader)
        accuracy = 100. * correct / total

        class_names = data_info["classes"]
        report = classification_report(all_targets, all_preds,
                                     target_names=class_names,
                                     output_dict=True, zero_division=0)

        cm = confusion_matrix(all_targets, all_preds)

        metrics = {
            "test_accuracy": accuracy,
            "test_loss": test_loss,
            "total_samples": total,
            "correct_predictions": correct,
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1_score": report["macro avg"]["f1-score"],
            "confusion_matrix": cm.tolist(),
            "per_class_metrics": {name: report[name] for name in class_names if name in report}
        }

        ref = storage.save_json("_p", "_n", "metrics", metrics)
        return {"metrics": ref}

    def get_output_preview(self, outputs, storage):
        metrics = storage.load_json(outputs["metrics"])
        return {
            "test_accuracy": f"{metrics['test_accuracy']:.2f}%",
            "test_loss": round(metrics["test_loss"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1_score": round(metrics["f1_score"], 4),
            "total_test_samples": metrics["total_samples"]
        }
