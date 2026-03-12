import os
import cv2
import yaml
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.data_utils import get_dataloaders
from src.evaluation.metrics import calculate_classification_metrics, save_metrics

from src.features.texture import extract_features

def load_and_extract(file_paths, labels, config):
    X = []
    Y = []
    for fp, lbl in tqdm(zip(file_paths, labels), total=len(file_paths), desc="Extracting Classical Features"):
        feat = extract_features(fp, config)
        if feat is not None:
            X.append(feat)
            Y.append(lbl)
    return np.array(X), np.array(Y)

def run_texture_svm(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    print("Loading data splits...")
    _, (X_train_paths, y_train, X_val_paths, y_val, X_test_paths, y_test) = get_dataloaders(config)
    
    print(f"Training set: {len(X_train_paths)} images")
    X_train_feat, Y_train = load_and_extract(X_train_paths, y_train, config)
    X_test_feat, Y_test = load_and_extract(X_test_paths, y_test, config)
    
    print("Training SVM Pipeline...")
    # Using pipeline for scaling and training
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=config["training"]["seed"]))
    clf.fit(X_train_feat, Y_train)
    
    print("Evaluating...")
    preds = clf.predict(X_test_feat)
    metrics = calculate_classification_metrics(Y_test, preds)
    
    print("Pipeline 1 Results:")
    for k, v in metrics.items():
        print(f"  {k.capitalize()}: {v:.4f}")
        
    # Save model and results
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    with open("models/texture_svm.pkl", "wb") as f:
        pickle.dump(clf, f)
        
    save_metrics(metrics, "results/pipeline1_metrics.json")
    np.savez("results/pipeline1_predictions.npz", y_true=Y_test, y_pred=preds)
    print("Saved Pipeline 1 outputs.")

if __name__ == "__main__":
    run_texture_svm()
