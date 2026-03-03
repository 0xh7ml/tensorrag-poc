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

        base_transforms = [transforms.Resize((data_info["resize_dim"], data_info["resize_dim"]))]

        aug_transforms = []
        if config.get("horizontal_flip", True):
            aug_transforms.append(transforms.RandomHorizontalFlip(p=0.5))

        rotation = config.get("rotation", 15)
        if rotation > 0:
            aug_transforms.append(transforms.RandomRotation(rotation))

        if config.get("color_jitter", True):
            aug_transforms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))

        final_transforms = [transforms.ToTensor()]
        if data_info["normalized"]:
            final_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225]))

        train_transform = transforms.Compose(base_transforms + aug_transforms + final_transforms)
        test_transform = transforms.Compose(base_transforms + final_transforms)

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
            applied.append(f"Rotation +/-{augs['rotation_degrees']} deg")
        if augs["color_jitter"]:
            applied.append("Color Jitter")

        return {
            "augmentations_applied": len(applied),
            "techniques": ", ".join(applied) if applied else "None",
            "training_samples": info["train_size"],
            "test_samples": info["test_size"]
        }
