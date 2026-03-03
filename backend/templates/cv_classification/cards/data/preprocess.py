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

        data_root = dataset_info["data_root"]
        dataset_type = dataset_info["dataset_type"]

        if dataset_type == "CIFAR10":
            train_dataset = datasets.CIFAR10(data_root, train=True, transform=transform)
            test_dataset = datasets.CIFAR10(data_root, train=False, transform=transform)
        elif dataset_type == "CIFAR100":
            train_dataset = datasets.CIFAR100(data_root, train=True, transform=transform)
            test_dataset = datasets.CIFAR100(data_root, train=False, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        sample_batch = next(iter(train_loader))
        sample_images, sample_labels = sample_batch

        preprocessed_info = {
            **dataset_info,
            "resize_dim": resize_dim,
            "normalized": normalize,
            "final_shape": list(sample_images.shape[1:]),
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
