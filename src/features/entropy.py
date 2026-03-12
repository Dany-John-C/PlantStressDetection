import cv2
import numpy as np
from skimage.feature import graycomatrix

def extract_glcm_entropy(image_path, config):
    img = cv2.imread(image_path)
    if img is None: return None
    
    img = cv2.resize(img, tuple(config["dataset"]["image_size"]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    distances = config["texture_svm"]["glcm_distances"]
    angles = config["texture_svm"]["glcm_angles"]
    
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    entropies = []
    for i in range(glcm.shape[2]):
        for j in range(glcm.shape[3]):
            p = glcm[:, :, i, j]
            p_non_zero = p[p > 0]
            entropy = -np.sum(p_non_zero * np.log2(p_non_zero))
            entropies.append(entropy)
            
    return np.array(entropies)
