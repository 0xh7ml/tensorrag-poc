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
            raise ValueError(f"Dataset type {dataset_type} not supported yet")

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
