import os
import yaml
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import cv2

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.data_utils import PlantDataset, get_dataloaders
from src.evaluation.metrics import save_metrics
from src.visualization.plot_style import apply_dark_style, GREEN, BLUE, YELLOW, RED, TEXT, MUTED, BORDER
apply_dark_style()

def get_robust_dataloader(X_test, y_test, aug_type, config):
    image_size = config["dataset"]["image_size"]
    
    if aug_type == "baseline":
        transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    elif aug_type == "blur":
        transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.GaussianBlur(blur_limit=(7, 11), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    elif aug_type == "noise":
        transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.GaussNoise(var_limit=(100.0, 200.0), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    elif aug_type == "brightness_low":
        transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.RandomBrightnessContrast(brightness_limit=(-0.4, -0.4), contrast_limit=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
    dataset = PlantDataset(X_test, y_test, transform=transform)
    return DataLoader(dataset, batch_size=config["dataset"]["batch_size"], num_workers=config["dataset"]["num_workers"], pin_memory=True)

def evaluate_robustness(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return accuracy_score(all_labels, all_preds)

def run_robustness_study(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    loaders, (_, _, _, _, X_test, y_test) = get_dataloaders(config)
    num_classes = loaders['num_classes']
    
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    model_path = "models/mobilenet_classifier.pth"
    if not os.path.exists(model_path):
        print("Please run Pipeline 2 (MobileNet Classifier) first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    conditions = ["baseline", "blur", "noise", "brightness_low"]
    results = {}
    
    print("Running Robustness Evaluation...")
    for cond in tqdm(conditions, desc="Perturbations"):
        loader = get_robust_dataloader(X_test, y_test, cond, config)
        acc = evaluate_robustness(model, loader, device)
        results[cond] = float(acc)
        print(f"  {cond.capitalize()}: Acc = {acc:.4f}")
        
    os.makedirs("results/studies", exist_ok=True)
    
    baseline = results.get("baseline", 1.0)
    bar_colors = [GREEN if k == "baseline" else (YELLOW if results[k] > 0.5 else RED) for k in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([k.replace("_", " ").title() for k in results], results.values(),
                  color=bar_colors, width=0.5)
    ax.axhline(baseline, color=MUTED, linestyle='--', linewidth=1.5, label=f'Baseline ({baseline:.3f})')
    for bar, v in zip(bars, results.values()):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f"{v:.3f}", ha='center', va='bottom', fontsize=10, color=TEXT)
    ax.set_title('MobileNetV3 Robustness Under Perturbations', fontsize=13)
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig("results/studies/robustness.png")
    plt.close()
    
    save_metrics(results, "results/studies/robustness_metrics.json")

if __name__ == "__main__":
    run_robustness_study()
