# Computer Vision Image Classification Pipeline

Build a complete image classification pipeline using 8 individual cards. Each card handles one step, and data flows between them via S3 storage.

## Pipeline Overview

```
Image Load → Preprocess → Data Augmentation → Build CNN Model → Train Model → Evaluate → Inference → Grad-CAM Visualization
```

## Project File Structure

Create the following folders and files in the **Editor** view:

```
cv-classification/            ← Project name
├── data/                     ← Folder
│   ├── image_load.py         ← Card 1
│   ├── preprocess.py         ← Card 2
│   └── augmentation.py       ← Card 3
├── model/                    ← Folder
│   └── build_cnn.py          ← Card 4
├── training/                 ← Folder
│   └── train_model.py        ← Card 5
├── evaluation/               ← Folder
│   └── evaluate.py           ← Card 6
├── inference/                ← Folder
│   └── inference.py          ← Card 7
└── visualization/            ← Folder
    └── grad_cam.py          ← Card 8
```

## Card Connection Map

| # | Card | File | Folder | Receives from | Sends to |
|---|------|------|--------|--------------|----------|
| 1 | Image Load | `image_load.py` | `data/` | — (config: dataset path) | `raw_dataset` |
| 2 | Preprocess | `preprocess.py` | `data/` | `raw_dataset` | `preprocessed_data` |
| 3 | Data Augmentation | `augmentation.py` | `data/` | `preprocessed_data` | `augmented_data` |
| 4 | Build CNN Model | `build_cnn.py` | `model/` | `augmented_data` | `model_config` |
| 5 | Train Model | `train_model.py` | `training/` | `model_config`, `augmented_data` | `trained_model` |
| 6 | Evaluate | `evaluate.py` | `evaluation/` | `trained_model`, `preprocessed_data` | `metrics` |
| 7 | Inference | `inference.py` | `inference/` | `trained_model` | `predictions` |
| 8 | Grad-CAM | `grad_cam.py` | `visualization/` | `trained_model` | `heatmaps` |

---

## Card 1: Image Load

**File:** `image_load.py` | **Folder:** `data/`

Loads image dataset from directory structure or downloads a standard dataset.

```python
from cards.base import BaseCard
import os
import torch
import torchvision
from torchvision import datasets
from PIL import Image
import json

class ImageLoadCard(BaseCard):
    card_type = "cv_image_load"
    display_name = "Image Load"
    description = "Load image dataset"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "dataset_type": {
            "type": "string", 
            "label": "Dataset type",
            "default": "CIFAR10"
        },
        "data_root": {
            "type": "string",
            "label": "Data root directory", 
            "default": "/tmp/cv_data"
        },
        "download": {
            "type": "boolean",
            "label": "Download dataset if missing",
            "default": True
        }
    }
    input_schema = {}
    output_schema = {"raw_dataset": "json"}

    def execute(self, config, inputs, storage):
        dataset_type = config.get("dataset_type", "CIFAR10")
        data_root = config.get("data_root", "/tmp/cv_data")
        download = config.get("download", True)
        
        os.makedirs(data_root, exist_ok=True)
        
        if dataset_type == "CIFAR10":
            train_dataset = datasets.CIFAR10(data_root, train=True, download=download)
            test_dataset = datasets.CIFAR10(data_root, train=False, download=download)
            classes = train_dataset.classes
            num_classes = len(classes)
            image_shape = (32, 32, 3)
        elif dataset_type == "CIFAR100":
            train_dataset = datasets.CIFAR100(data_root, train=True, download=download)
            test_dataset = datasets.CIFAR100(data_root, train=False, download=download)
            classes = train_dataset.classes
            num_classes = len(classes)
            image_shape = (32, 32, 3)
        else:
            # For custom datasets, implement directory loading
            raise ValueError(f"Dataset type {dataset_type} not supported yet")
        
        # Store metadata
        dataset_info = {
            "dataset_type": dataset_type,
            "data_root": data_root,
            "classes": classes,
            "num_classes": num_classes,
            "image_shape": image_shape,
            "train_size": len(train_dataset),
            "test_size": len(test_dataset)
        }
        
        ref = storage.save_json("_p", "_n", "raw_dataset", dataset_info)
        return {"raw_dataset": ref}

    def get_output_preview(self, outputs, storage):
        info = storage.load_json(outputs["raw_dataset"])
        return {
            "columns": ["Property", "Value"],
            "rows": [
                ["Dataset", info["dataset_type"]],
                ["Classes", info["num_classes"]],
                ["Image Shape", f"{info['image_shape'][0]}x{info['image_shape'][1]}x{info['image_shape'][2]}"],
                ["Train Images", info["train_size"]],
                ["Test Images", info["test_size"]],
                ["Class Names", ", ".join(info["classes"][:5]) + "..." if len(info["classes"]) > 5 else ", ".join(info["classes"])]
            ],
            "total_rows": 6
        }
```

## Card 2: Preprocess

**File:** `preprocess.py` | **Folder:** `data/`

Preprocesses images with normalization and resizing.

```python
from cards.base import BaseCard
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import numpy as np

class PreprocessCard(BaseCard):
    card_type = "cv_preprocess"
    display_name = "Preprocess Images"
    description = "Normalize and resize images"
    category = "data"
    execution_mode = "local"
    output_view_type = "metrics"

    config_schema = {
        "resize_dim": {
            "type": "number",
            "label": "Resize to dimension (square)",
            "default": 224
        },
        "normalize": {
            "type": "boolean", 
            "label": "Apply ImageNet normalization",
            "default": True
        }
    }
    input_schema = {"raw_dataset": "json"}
    output_schema = {"preprocessed_data": "json"}

    def execute(self, config, inputs, storage):
        dataset_info = storage.load_json(inputs["raw_dataset"])
        resize_dim = int(config.get("resize_dim", 224))
        normalize = config.get("normalize", True)
        
        # Build transforms
        transform_list = [
            transforms.Resize((resize_dim, resize_dim))
        ]
        
        if normalize:
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            transform_list.append(transforms.ToTensor())
            
        transform = transforms.Compose(transform_list)
        
        # Load datasets with transforms
        data_root = dataset_info["data_root"]
        dataset_type = dataset_info["dataset_type"]
        
        if dataset_type == "CIFAR10":
            train_dataset = datasets.CIFAR10(data_root, train=True, transform=transform)
            test_dataset = datasets.CIFAR10(data_root, train=False, transform=transform)
        elif dataset_type == "CIFAR100":
            train_dataset = datasets.CIFAR100(data_root, train=True, transform=transform)
            test_dataset = datasets.CIFAR100(data_root, train=False, transform=transform)
        
        # Create data loaders for sampling
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Compute dataset statistics
        sample_batch = next(iter(train_loader))
        sample_images, sample_labels = sample_batch
        
        preprocessed_info = {
            **dataset_info,
            "resize_dim": resize_dim,
            "normalized": normalize,
            "final_shape": list(sample_images.shape[1:]),  # [C, H, W]
            "transform_applied": str(transform),
            "sample_tensor_shape": list(sample_images.shape),
            "sample_mean": float(sample_images.mean()),
            "sample_std": float(sample_images.std())
        }
        
        ref = storage.save_json("_p", "_n", "preprocessed_data", preprocessed_info)
        return {"preprocessed_data": ref}

    def get_output_preview(self, outputs, storage):
        info = storage.load_json(outputs["preprocessed_data"])
        return {
            "original_shape": f"{info['image_shape'][0]}x{info['image_shape'][1]}",
            "final_shape": f"{info['final_shape'][1]}x{info['final_shape'][2]}",
            "channels": info['final_shape'][0],
            "normalized": "Yes" if info['normalized'] else "No",
            "sample_mean": round(info['sample_mean'], 4),
            "sample_std": round(info['sample_std'], 4)
        }
```

## Card 3: Data Augmentation

**File:** `augmentation.py` | **Folder:** `data/`

Applies data augmentation techniques to increase dataset diversity.

```python
from cards.base import BaseCard
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

class DataAugmentationCard(BaseCard):
    card_type = "cv_augmentation"
    display_name = "Data Augmentation" 
    description = "Apply data augmentation to training set"
    category = "data"
    execution_mode = "local"
    output_view_type = "metrics"

    config_schema = {
        "horizontal_flip": {
            "type": "boolean",
            "label": "Random horizontal flip",
            "default": True
        },
        "rotation": {
            "type": "number", 
            "label": "Random rotation (degrees)",
            "default": 15
        },
        "color_jitter": {
            "type": "boolean",
            "label": "Color jittering",
            "default": True
        }
    }
    input_schema = {"preprocessed_data": "json"}
    output_schema = {"augmented_data": "json"}

    def execute(self, config, inputs, storage):
        data_info = storage.load_json(inputs["preprocessed_data"])
        
        # Base transforms (same as preprocessing)
        base_transforms = [transforms.Resize((data_info["resize_dim"], data_info["resize_dim"]))]
        
        # Augmentation transforms for training
        aug_transforms = []
        if config.get("horizontal_flip", True):
            aug_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
        
        rotation = config.get("rotation", 15)
        if rotation > 0:
            aug_transforms.append(transforms.RandomRotation(rotation))
            
        if config.get("color_jitter", True):
            aug_transforms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
        
        # Final transforms
        final_transforms = [transforms.ToTensor()]
        if data_info["normalized"]:
            final_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                       std=[0.229, 0.224, 0.225]))
        
        # Training transform with augmentation
        train_transform = transforms.Compose(base_transforms + aug_transforms + final_transforms)
        
        # Test transform (no augmentation)
        test_transform = transforms.Compose(base_transforms + final_transforms)
        
        # Create datasets
        data_root = data_info["data_root"]
        dataset_type = data_info["dataset_type"]
        
        if dataset_type == "CIFAR10":
            train_dataset = datasets.CIFAR10(data_root, train=True, transform=train_transform)
            test_dataset = datasets.CIFAR10(data_root, train=False, transform=test_transform)
        elif dataset_type == "CIFAR100":
            train_dataset = datasets.CIFAR100(data_root, train=True, transform=train_transform)
            test_dataset = datasets.CIFAR100(data_root, train=False, transform=test_transform)
        
        augmented_info = {
            **data_info,
            "augmentations": {
                "horizontal_flip": config.get("horizontal_flip", True),
                "rotation_degrees": config.get("rotation", 15),
                "color_jitter": config.get("color_jitter", True)
            },
            "train_transform": str(train_transform),
            "test_transform": str(test_transform)
        }
        
        ref = storage.save_json("_p", "_n", "augmented_data", augmented_info)
        return {"augmented_data": ref}

    def get_output_preview(self, outputs, storage):
        info = storage.load_json(outputs["augmented_data"])
        augs = info["augmentations"]
        
        applied = []
        if augs["horizontal_flip"]:
            applied.append("Horizontal Flip")
        if augs["rotation_degrees"] > 0:
            applied.append(f"Rotation ±{augs['rotation_degrees']}°")
        if augs["color_jitter"]:
            applied.append("Color Jitter")
            
        return {
            "augmentations_applied": len(applied),
            "techniques": ", ".join(applied) if applied else "None",
            "training_samples": info["train_size"],
            "test_samples": info["test_size"]
        }
```

## Card 4: Build CNN Model

**File:** `build_cnn.py` | **Folder:** `model/`

Creates a CNN architecture (ResNet, EfficientNet, or custom).

```python
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
        
        # Build model
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
            # Custom simple CNN
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
        
        # Count parameters
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
```

## Card 5: Train Model

**File:** `train_model.py` | **Folder:** `training/`

Trains the CNN model with the augmented dataset.

```python
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
        
        # Rebuild model
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
            # Custom CNN (same as in build_cnn.py)
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
        
        # Load saved weights
        state_dict = {k: torch.tensor(v) for k, v in model_info["model_state"].items()}
        model.load_state_dict(state_dict)
        model = model.to(device)
        
        # Build datasets with transforms
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
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=model_info["learning_rate"])
        
        # Training loop
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
        
        # Save trained model
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
```

## Card 6: Evaluate

**File:** `evaluate.py` | **Folder:** `evaluation/`

Evaluates the trained model on test data.

```python
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
        
        # Rebuild and load trained model  
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
            # Custom CNN
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
        
        # Load trained weights
        state_dict = {k: torch.tensor(v) for k, v in model_info["trained_state"].items()}
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        # Prepare test data
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
        
        # Evaluation
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
        
        # Detailed metrics
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
```

## Card 7: Inference

**File:** `inference.py` | **Folder:** `inference/`

Makes predictions on new images.

```python
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
        
        # Rebuild model
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
            # Custom CNN
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
        
        # Load weights
        state_dict = {k: torch.tensor(v) for k, v in model_info["trained_state"].items()}
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        # Load and preprocess image
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
        
        # Prediction
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
        
        # Show top 3 predictions
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
```

## Card 8: Grad-CAM Visualization

**File:** `grad_cam.py` | **Folder:** `visualization/`

Generates heatmaps showing what parts of the image the model focuses on.

```python
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
    category = "visualization"
    execution_mode = "local"
    output_view_type = "image"

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
        
        # Rebuild model
        arch = model_info["architecture"]
        num_classes = model_info["num_classes"]
        
        if arch == "resnet18":
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            target_layer = model.layer4[-1].conv2  # Last conv in layer4
        elif arch == "resnet50":
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            target_layer = model.layer4[-1].conv3  # Last conv in layer4
        else:
            # For other architectures, would need specific layer mapping
            raise ValueError(f"Grad-CAM not implemented for {arch}")
        
        # Load weights
        state_dict = {k: torch.tensor(v) for k, v in model_info["trained_state"].items()}
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        # Load image
        if image_url.startswith("http"):
            response = requests.get(image_url)
            original_image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            original_image = Image.open(image_url).convert('RGB')
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(original_image).unsqueeze(0).to(device)
        input_tensor.requires_grad_()
        
        # Hook to capture gradients and activations
        gradients = []
        activations = []
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        handle_backward = target_layer.register_backward_hook(backward_hook)  
        handle_forward = target_layer.register_forward_hook(forward_hook)
        
        # Forward pass
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
        
        # Backward pass for predicted class
        model.zero_grad()
        output[0, predicted_class].backward()
        
        # Generate Grad-CAM
        grad = gradients[0].squeeze()  # [C, H, W]
        activation = activations[0].squeeze()  # [C, H, W]
        
        weights = torch.mean(grad, dim=[1, 2])  # [C]
        cam = torch.sum(weights.unsqueeze(1).unsqueeze(2) * activation, dim=0)  # [H, W]
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        
        # Resize to input size and convert to numpy
        cam_resized = nn.functional.interpolate(
            cam.unsqueeze(0).unsqueeze(0), 
            size=(224, 224), 
            mode='bilinear'
        ).squeeze().cpu().numpy()
        
        # Create heatmap overlay
        original_resized = np.array(original_image.resize((224, 224)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = heatmap * 0.4 + original_resized * 0.6
        overlay = np.uint8(overlay)
        
        # Convert to base64 for storage
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
        
        # Cleanup hooks
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
```

---

## How to Wire the Pipeline

### Canvas Connections:

```
[Image Load] ────────────> [Preprocess] ────────────> [Data Augmentation]
                                │                            │
                                │                            │
                                └─────> [Evaluate] <─────────┘
                                               │              │
[Build CNN] ────────────────────────────> [Train Model] ────┤
     │                                            │          │  
     │                                            │          │
     └────> [Inference] <─────────────────────────┤          │
     │           │                                │          │
     │           │                                │          │
     └────> [Grad-CAM] <───────────────────────────┘          │
                                                              │
                                                         [Metrics]
```

### Key Configuration:

- **Image Load**: Choose CIFAR10/CIFAR100 or custom dataset
- **Build CNN**: Select ResNet18, ResNet50, EfficientNet-B0, or custom
- **Train Model**: Set epochs (10-50) and batch size (16-64)
- **Inference/Grad-CAM**: Provide image URLs for testing

This pipeline provides end-to-end computer vision capabilities from data loading through model interpretation with Grad-CAM visualizations.