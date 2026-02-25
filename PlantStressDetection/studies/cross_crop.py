import os
import yaml
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils import get_dataloaders
from utils.metrics import save_metrics

def evaluate_cross_crop(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Cross-Crop Evaluation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, acc

def run_cross_crop_study(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get regular loaders to determine total number of classes the model expects
    base_loaders, _ = get_dataloaders(config, cross_crop=True)
    num_classes = base_loaders['num_classes']
    
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    model_path = "models/mobilenet_classifier.pth"
    if not os.path.exists(model_path):
        print("Please run Pipeline 2 (MobileNet Classifier) first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    print("Evaluating on Unseen Crops...")
    # Now create dataloader for the unseen test crops
    # Notice we override config dynamically to load only test crops
    orig_train = config["cross_crop"]["train_crops"]
    config["cross_crop"]["train_crops"] = config["cross_crop"]["test_crops"]
    
    try:
        test_crop_loaders, _ = get_dataloaders(config, cross_crop=True)
        if len(test_crop_loaders['test'].dataset) == 0:
            print("No data found for unseen crops. Skipping cross-crop study.")
            return
            
        unseen_loss, unseen_acc = evaluate_cross_crop(model, test_crop_loaders['test'], device, criterion)
        
        # Reload train crops for baseline comparison
        config["cross_crop"]["train_crops"] = orig_train
        train_crop_loaders, _ = get_dataloaders(config, cross_crop=True)
        seen_loss, seen_acc = evaluate_cross_crop(model, train_crop_loaders['test'], device, criterion)
        
        results = {
            'Seen_Crops_Accuracy': float(seen_acc),
            'Unseen_Crops_Accuracy': float(unseen_acc),
            'Generalization_Degradation': float(seen_acc - unseen_acc)
        }
        
        print("\nCross-Crop Generalization Results:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
            
        os.makedirs("results/studies", exist_ok=True)
        save_metrics(results, "results/studies/cross_crop_metrics.json")
        
    except Exception as e:
        print(f"Cross-crop study failed (likely due to missing dummy data for unseen crops): {e}")

if __name__ == "__main__":
    run_cross_crop_study()
