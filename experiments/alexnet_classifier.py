import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import json

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.visualization.plot_style import apply_dark_style, PALETTE, GREEN, BLUE, YELLOW, RED, DARK_BG, PANEL_BG, BORDER, TEXT, MUTED
apply_dark_style()

from src.preprocessing.data_utils import get_dataloaders
from src.evaluation.metrics import calculate_classification_metrics, save_metrics

from src.models.alexnet import AlexNetKerasPort, freeze_bottom_layers

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_classification_metrics(all_labels, all_preds)
    
    return epoch_loss, metrics

def evaluate(model, dataloader, criterion, device, return_probs=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if return_probs:
                all_probs.extend(probs.cpu().numpy())
                
    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_classification_metrics(all_labels, all_preds)
    
    if return_probs:
        return epoch_loss, metrics, all_labels, all_preds, np.array(all_probs)
    return epoch_loss, metrics

def plot_training_history(history, output_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['train_loss'], color=BLUE,  label='Train',      linewidth=2)
    ax1.plot(epochs, history['val_loss'],   color=GREEN, label='Validation', linewidth=2, linestyle='--')
    ax1.fill_between(epochs, history['train_loss'], history['val_loss'], alpha=0.08, color=YELLOW)
    ax1.set_title('Loss over Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)

    train_acc = [m['accuracy'] for m in history['train_metrics']]
    val_acc   = [m['accuracy'] for m in history['val_metrics']]
    ax2.plot(epochs, train_acc, color=BLUE,  label='Train',      linewidth=2)
    ax2.plot(epochs, val_acc,   color=GREEN, label='Validation', linewidth=2, linestyle='--')
    ax2.fill_between(epochs, train_acc, val_acc, alpha=0.08, color=GREEN)
    ax2.set_title('Accuracy over Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True)

    plt.suptitle('AlexNet Training History', fontsize=15, color=TEXT, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 1.2), max(7, len(class_names))))
    cmap = sns.color_palette("light:#3fb950", as_cmap=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor=BORDER, ax=ax,
                annot_kws={"size": 9, "color": DARK_BG})
    ax.set_ylabel('Actual',    fontsize=11)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_title('Confusion Matrix', fontsize=13)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(y_true, y_probs, num_classes, class_names, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    y_true_bin = np.zeros((len(y_true), num_classes))
    for i, label in enumerate(y_true):
        y_true_bin[i, label] = 1

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        color = PALETTE[i % len(PALETTE)]
        ax.plot(fpr, tpr, lw=2, color=color, label=f'{class_names[i]} (AUC={roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.5, linestyle='--', label='Random')
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color=MUTED)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (One-vs-Rest)')
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_alexnet_pipeline(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    loaders, _ = get_dataloaders(config)
    num_classes = loaders['num_classes']
    class_to_idx = loaders['class_to_idx']
    idx_to_class = {v: k.split('___')[1] if '___' in k else k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]
    
    model = AlexNetKerasPort(num_classes=num_classes)
    
    # Optional: Load weights if they exist and we're not fine-tuning
    # We freeze layers as requested by the original poster
    model = freeze_bottom_layers(model)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_metrics': [], 'val_metrics': []
    }
    
    epochs = config["training"]["epochs"]
    
    os.makedirs("results/alexnet", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # In the reference code, SGD is used with lr=0.001, momentum=0.9, decay=0.005.
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config["alexnet"]["learning_rate"], 
        momentum=config["alexnet"]["momentum"], 
        weight_decay=config["alexnet"]["weight_decay"]
    )
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        t_loss, t_metrics = train_one_epoch(model, loaders['train'], criterion, optimizer, device)
        v_loss, v_metrics = evaluate(model, loaders['valid'], criterion, device)
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_metrics'].append(t_metrics)
        history['val_metrics'].append(v_metrics)
        
        print(f"  Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_metrics['accuracy']:.4f}")
        
    print("\nEvaluating on Test Set...")
    test_loss, test_metrics, y_true, y_pred, y_probs = evaluate(
        model, loaders['test'], criterion, device, return_probs=True
    )
    
    print("\nPipeline 6 Test Results (AlexNet):")
    for k, v in test_metrics.items():
        print(f"  {k.capitalize()}: {v:.4f}")
        
    plot_training_history(history, "results/alexnet/training_history.png")
    plot_confusion_matrix(y_true, y_pred, class_names, "results/alexnet/confusion_matrix.png")
    plot_roc_curve(y_true, y_probs, num_classes, class_names, "results/alexnet/roc_curve.png")
    
    save_metrics(test_metrics, "results/alexnet/test_metrics.json")
    torch.save(model.state_dict(), "models/alexnet_classifier.pth")
    print("Pipeline 6 completed. Artifacts saved in results/alexnet/")

if __name__ == "__main__":
    run_alexnet_pipeline()
