import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

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
    
    # 3. Color Histogram HSV
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
