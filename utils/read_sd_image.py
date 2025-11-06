import os
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms as transforms
import glob
from torchvision.datasets import ImageFolder
import torch

class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        """
        Custom dataset for loading image files.
        :param image_paths: List of image file paths
        :param transform: Image preprocessing transform
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except FileNotFoundError:
            print(f"Warning: File {img_path} not found, using default image")
            default_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                default_image = self.transform(default_image)
            return default_image

def sanitize_label(label):
    """Remove invalid characters from labels (e.g., newlines, special chars)."""
    return label.replace("/", "_").replace("\n", "").replace("\"", "").strip()

def load_real_ood_images(real_ood_path, preprocess, batch_size=32):
    """
    Load real OOD images from standard benchmarks.
    :param real_ood_path: Root directory of real OOD images
    :param preprocess: Image preprocessing function
    :param batch_size: Batch size
    :return: List of real OOD datasets and total image count
    """
    real_ood_datasets = []
    total_images = 0

    if not os.path.exists(real_ood_path):
        print(f"Warning: Real OOD image path {real_ood_path} does not exist")
        return real_ood_datasets, total_images

    ood_datasets = ['iNaturalist', 'SUN', 'places365', 'dtd']
    for dataset_name in ood_datasets:
        dataset_path = os.path.join(real_ood_path, dataset_name)
        if os.path.exists(dataset_path):
            image_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_paths.extend(glob.glob(os.path.join(dataset_path, ext)))

            if image_paths:
                dataset = CustomDataset(image_paths, transform=preprocess)
                real_ood_datasets.append(dataset)
                total_images += len(image_paths)
            else:
                print(f"Warning: No images found in {dataset_path}")

    return real_ood_datasets, total_images

def load_data_by_batches(label_list, data_dir, preprocess, batch_size=32, include_real_ood=False, real_ood_path="/data0/fayi/sampled_ood_images"):
    """
    Load generated images in three batches, optionally with real OOD images.
    :param label_list: List of label names
    :param data_dir: Root directory of images
    :param preprocess: Image preprocessing function
    :param batch_size: Batch size for DataLoader
    :param include_real_ood: Whether to include real OOD images
    :param real_ood_path: Root directory of real OOD images
    :return: Three DataLoader objects with combined synthetic and real OOD images
    """
    batch1_paths = []
    batch2_paths = []
    batch3_paths = []

    skipped_count = 0

    for folder_name in label_list:
        folder_name = sanitize_label(folder_name)
        folder_path = os.path.join(data_dir, folder_name)

        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist, skipping")
            skipped_count += 1
            continue

        img1_path = os.path.join(folder_path, f"{folder_name}_1.png")
        img2_path = os.path.join(folder_path, f"{folder_name}_2.png")
        img3_path = os.path.join(folder_path, f"{folder_name}_3.png")

        if os.path.exists(img1_path) or os.path.exists(img2_path) or os.path.exists(img3_path):
            batch1_paths.append(img1_path)
            batch2_paths.append(img2_path)
            batch3_paths.append(img3_path)
        else:
            print(f"Warning: No valid images found in folder {folder_path}, skipping")
            skipped_count += 1

    if skipped_count > 0:
        print(f"Skipped {skipped_count} label folders (missing or no valid images)")

    if len(batch1_paths) == 0:
        raise ValueError(f"No valid images found. Check data_dir: {data_dir} and label paths")

    dataset1 = CustomDataset(batch1_paths, transform=preprocess)
    dataset2 = CustomDataset(batch2_paths, transform=preprocess)
    dataset3 = CustomDataset(batch3_paths, transform=preprocess)

    real_ood_datasets = []
    real_ood_images_count = 0
    if include_real_ood:
        real_ood_datasets, real_ood_images_count = load_real_ood_images(real_ood_path, preprocess, batch_size)

    loader1 = DataLoader(dataset1, batch_size=batch_size, num_workers=2, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=batch_size, num_workers=2, shuffle=False)
    loader3 = DataLoader(dataset3, batch_size=batch_size, num_workers=2, shuffle=False)

    if real_ood_datasets:
        combined_dataset = ConcatDataset([dataset1] + real_ood_datasets)
        combined_loader = DataLoader(combined_dataset, batch_size=batch_size, num_workers=2, shuffle=False)
    else:
        combined_loader = loader1

    return combined_loader, loader2, loader3
