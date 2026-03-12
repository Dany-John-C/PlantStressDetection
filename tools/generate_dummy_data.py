import os
import numpy as np
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm

def create_dummy_dataset(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    data_dir = config["dataset"]["original_data_dir"]
    os.makedirs(data_dir, exist_ok=True)
    
    # Simulate a subset of classes for testing
    classes = [
        "Tomato___healthy",
        "Tomato___Bacterial_spot",
        "Potato___healthy",
        "Potato___Early_blight",
        "Pepper,_bell___healthy",
        "Pepper,_bell___Bacterial_spot",
        "Apple___healthy",
        "Apple___Apple_scab"
    ]
    
    images_per_class = 50 # Small number for dry run
    image_size = config["dataset"]["image_size"]
    
    print(f"Generating dummy dataset in {data_dir}...")
    
    for cls in tqdm(classes, desc="Generating Classes"):
        cls_dir = os.path.join(data_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        # Decide if healthy or diseased based on name
        is_healthy = "healthy" in cls.lower()
        
        for i in range(images_per_class):
            # Create a simple background (greenish leaf color)
            img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
            img[:] = (34, 139, 34) # Forest green
            
            # Add some simulated texture/noise
            noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
            
            # If diseased, simulate some lesions
            if not is_healthy:
                num_lesions = np.random.randint(1, 10)
                for _ in range(num_lesions):
                    center_x = np.random.randint(20, image_size[1] - 20)
                    center_y = np.random.randint(20, image_size[0] - 20)
                    axes = (np.random.randint(5, 20), np.random.randint(5, 20))
                    angle = np.random.randint(0, 360)
                    # Brown/dark lesions
                    color = (19, 69, 139) # BGR
                    cv2.ellipse(img, (center_x, center_y), axes, angle, 0, 360, color, -1)
            
            img_path = os.path.join(cls_dir, f"{cls}_img{i:03d}.jpg")
            cv2.imwrite(img_path, img)
            
    print("Dummy dataset generation complete.")

if __name__ == "__main__":
    create_dummy_dataset()
