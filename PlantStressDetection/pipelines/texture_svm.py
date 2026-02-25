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
from utils.data_utils import get_dataloaders
from utils.metrics import calculate_classification_metrics, save_metrics

def extract_features(image_path, config):
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    img = cv2.resize(img, tuple(config["dataset"]["image_size"]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    features = []
    
    # 1. GLCM Features
    distances = config["texture_svm"]["glcm_distances"]
    angles = config["texture_svm"]["glcm_angles"]
    
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    # Extract contrast, correlation, energy, homogeneity
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        features.extend(graycoprops(glcm, prop).flatten())
        
    # 2. Local Binary Patterns (LBP)
    radius = config["texture_svm"]["lbp_radius"]
    n_points = config["texture_svm"]["lbp_n_points"]
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    features.extend(hist)
    
    # 3. Color Histogram (HSV for robust color representation instead of just RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
    
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    
    features.extend(hist_h.flatten())
    features.extend(hist_s.flatten())
    features.extend(hist_v.flatten())
    
    return np.array(features)

def load_and_extract(file_paths, labels, config):
    X = []
    Y = []
    for fp, lbl in tqdm(zip(file_paths, labels), total=len(file_paths), desc="Extracting Classical Features"):
        feat = extract_features(fp, config)
        if feat is not None:
            X.append(feat)
            Y.append(lbl)
    return np.array(X), np.array(Y)

def run_texture_svm(config_path="config.yaml"):
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
    print("Saved Pipeline 1 outputs.")

if __name__ == "__main__":
    run_texture_svm()
