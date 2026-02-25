import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PlantDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, return_proxy=False):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.return_proxy = return_proxy

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        if self.return_proxy:
            # We will compute the lesion proxy on the fly or load it
            # For simplicity, if requested, compute basic proxy from original image
            proxy = compute_lesion_proxy(cv2.imread(img_path))
            return image, label, torch.tensor(proxy, dtype=torch.float32)

        return image, label

def compute_lesion_proxy(image):
    """
    Early-Stage Proxy Creation
    Estimates lesion coverage percentage by analyzing color anomalies (non-green parts).
    Returns severity score (0.0 to 1.0)
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for healthy green
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    # Mask of green areas
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Total plant area (assuming non-black background after some basic thresholding)
    # Since dataset often has diverse backgrounds, a simple proxy:
    # Everything that is NOT green and NOT pure background is lesion.
    # We will assume a simple thresholding for the leaf contour
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, leaf_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    
    leaf_area = cv2.countNonZero(leaf_mask)
    if leaf_area == 0:
        return 0.0
        
    green_area = cv2.countNonZero(mask_green)
    lesion_area = leaf_area - green_area
    
    # Clamp to > 0
    lesion_area = max(0, lesion_area)
    coverage = lesion_area / leaf_area
    return min(1.0, coverage)

def get_augmentations(image_size):
    """
    Robustness Analysis Augmentations
    Brightness, Blur, Noise, Clutter simulation
    """
    return {
        'train': A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2), # Brightness variation
            A.GaussianBlur(blur_limit=3, p=0.2), # Blur
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2), # Noise
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        'valid': A.Compose([
            A.Resize(image_size[0], image_size[1]),
             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        'robustness': A.Compose([
            A.Resize(image_size[0], image_size[1]),
            # Apply severe perturbations for robustness test
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=1.0),
            A.GaussianBlur(blur_limit=(5, 7), p=1.0),
            A.GaussNoise(var_limit=(50.0, 100.0), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    }

def get_dataloaders(config, cross_crop=False):
    """
    Loads dataloaders ensuring no leakage by properly splitting files, then classes.
    """
    data_dir = config["dataset"]["original_data_dir"]
    all_classes = sorted(os.listdir(data_dir))
    
    file_paths = []
    labels = []
    
    # Simple mapping
    class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
    
    for cls in all_classes:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir): continue
        
        # Determine if we filter by crop for cross-crop
        if cross_crop:
            train_crops = config["cross_crop"]["train_crops"]
            if not any(crop in cls for crop in train_crops):
                continue
                
        paths = glob.glob(os.path.join(cls_dir, "*.jpg"))
        paths.extend(glob.glob(os.path.join(cls_dir, "*.JPG")))
        
        file_paths.extend(paths)
        labels.extend([class_to_idx[cls]] * len(paths))
        
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        file_paths, labels, test_size=1.0 - config["dataset"]["splits"]["train"], 
        random_state=config["training"]["seed"], stratify=labels
    )
    
    val_size = config["dataset"]["splits"]["valid"] / (config["dataset"]["splits"]["valid"] + config["dataset"]["splits"]["test"])
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1.0 - val_size,
        random_state=config["training"]["seed"], stratify=y_temp
    )
    
    augs = get_augmentations(config["dataset"]["image_size"])
    
    train_dataset = PlantDataset(X_train, y_train, transform=augs['train'])
    val_dataset = PlantDataset(X_val, y_val, transform=augs['valid'])
    test_dataset = PlantDataset(X_test, y_test, transform=augs['valid'])
    
    loader_args = {
        'batch_size': config["dataset"]["batch_size"],
        'num_workers': config["dataset"]["num_workers"],
        'pin_memory': True
    }
    
    loaders = {
        'train': DataLoader(train_dataset, shuffle=True, **loader_args),
        'valid': DataLoader(val_dataset, shuffle=False, **loader_args),
        'test': DataLoader(test_dataset, shuffle=False, **loader_args),
        'num_classes': len(all_classes),
        'class_to_idx': class_to_idx
    }
    return loaders, (X_train, y_train, X_val, y_val, X_test, y_test)
