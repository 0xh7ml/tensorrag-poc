from cards.base import BaseCard
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader

class TrainModelCard(BaseCard):
    card_type = "cv_train_model"
    display_name = "Train Model"
    description = "Train CNN with augmented data"
    category = "training"
    execution_mode = "local"
    output_view_type = "metrics"

    config_schema = {
        "epochs": {
            "type": "number",
            "label": "Number of epochs",
            "default": 10
        },
        "batch_size": {
            "type": "number",
            "label": "Batch size",
            "default": 32
        }
    }
    input_schema = {"model_config": "json", "augmented_data": "json"}
    output_schema = {"trained_model": "json"}

    def execute(self, config, inputs, storage):
        model_info = storage.load_json(inputs["model_config"])
        data_info = storage.load_json(inputs["augmented_data"])

        epochs = int(config.get("epochs", 10))
        batch_size = int(config.get("batch_size", 32))

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

        state_dict = {k: torch.tensor(v) for k, v in model_info["model_state"].items()}
        model.load_state_dict(state_dict)
        model = model.to(device)

        resize_dim = data_info["resize_dim"]
        train_transform = transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        data_root = data_info["data_root"]
        dataset_type = data_info["dataset_type"]

        if dataset_type == "CIFAR10":
            train_dataset = datasets.CIFAR10(data_root, train=True, transform=train_transform)
        elif dataset_type == "CIFAR100":
            train_dataset = datasets.CIFAR100(data_root, train=True, transform=train_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=model_info["learning_rate"])

        model.train()
        train_losses = []
        train_accuracies = []

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)

        trained_info = {
            **model_info,
            "epochs_trained": epochs,
            "batch_size": batch_size,
            "final_train_loss": train_losses[-1],
            "final_train_accuracy": train_accuracies[-1],
            "train_loss_history": train_losses,
            "train_acc_history": train_accuracies,
            "trained_state": {k: v.tolist() for k, v in model.state_dict().items()}
        }

        ref = storage.save_json("_p", "_n", "trained_model", trained_info)
        return {"trained_model": ref}

    def get_output_preview(self, outputs, storage):
        info = storage.load_json(outputs["trained_model"])
        return {
            "epochs": info["epochs_trained"],
            "final_train_loss": round(info["final_train_loss"], 4),
            "final_train_accuracy": f"{info['final_train_accuracy']:.2f}%",
            "architecture": info["architecture"],
            "batch_size": info["batch_size"],
            "status": "Training completed"
        }
