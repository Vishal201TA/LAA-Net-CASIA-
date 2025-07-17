from laa_datasets.builder import DATASETS
from torch.utils.data import Dataset
import os
import json
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

@DATASETS.register_module()
class CASIA2Dataset(Dataset):
    def __init__(self, config, split='train', **kwargs):
        self._cfg = config
        self.split = split
        self.train = split == 'train'
        self.samples = []

        split_cfg = config.DATA[split.upper()]
        self.image_dir = os.path.join(split_cfg.ROOT, 'images', split)
        self.mask_dir = os.path.join(split_cfg.ROOT, 'masks', split)
        self.label_folders = split_cfg.LABEL_FOLDER

        self.input_size = tuple(config.IMAGE_SIZE)
        self.output_size = tuple(config.HEATMAP_SIZE)

        self._load_data()

        # Apply transforms from cfg
        print("Resize input size:", self.input_size)

        self.img_transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.TRANSFORM.normalize.mean, std=config.TRANSFORM.normalize.std)
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(self.output_size),
            transforms.ToTensor()
        ])

    def _load_data(self):
        for label, folder in enumerate(self.label_folders):
            folder_path = os.path.join(self.image_dir, folder)
            for img_name in os.listdir(folder_path):
                sample = {
                    'image': os.path.join(folder_path, img_name),
                    'label': label,
                    'mask': os.path.join(self.mask_dir, folder, img_name) if label == 1 else None
                }
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")
        image = self.img_transform(image)
        label = sample["label"]
        cstency_heatmap = torch.zeros((1, self.output_size[0], self.output_size[1]))  # shape [1, 64, 64]


        if label == 1 and sample["mask"] and os.path.exists(sample["mask"]):
            mask = Image.open(sample["mask"]).convert("L")
            mask = self.mask_transform(mask)  # shape: [1, H, W]
        else:
            mask = torch.zeros((1, *self.output_size), dtype=torch.float32)

        # Repeat mask across channels to match model output (e.g., 2 channels)
        mask = mask.repeat(2, 1, 1)  # final shape: [2, H, W]


        # Return in dict format expected by the rest of the pipeline
        return {
            'img': image,
            'label': torch.tensor(label, dtype=torch.long),
            'target': mask,
            'heatmap': mask,
            'cstency_hm': mask,
            'offset': torch.zeros_like(mask),
            "cstency_heatmap": cstency_heatmap,
        }

    def train_worker_init_fn(self, worker_id):
        seed = torch.initial_seed() % 2**32
        np.random.seed(seed)
        random.seed(seed)

    def train_collate_fn(self, batch):
        collated = {}
        for key in batch[0]:
            collated[key] = torch.stack([item[key] for item in batch])
        return collated

