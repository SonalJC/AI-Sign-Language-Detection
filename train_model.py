import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# 1. Path to your data (Double check if 'Data' folder is exactly here)
DATA_DIR = r'C:\Users\Kishori\OneDrive\Documents\Ai Sign Language detection\archive (2)\Data'
data = []
labels = []

print("--- STEP 1: Starting Hybrid Data Extraction ---")

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path): continue
    
    print(f"Reading Letter: {dir_}")
    for img_path in os.listdir(dir_path)[:300]: 
        img = cv2.imread(os.path.join(dir_path, img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        # Use 64x64 Pixels (Kyuki Landmarks extract nahi ho rahe the)
        img_resized = cv2.resize(img, (64, 64)) 
        data.append(img_resized.flatten())
        labels.append(dir_)

# 2. Train and Save
if len(data) > 0:
    print(f"Training Model on {len(data)} images... Please wait.")
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(np.array(data), np.array(labels))
    joblib.dump(model, 'sign_language_model.p')
    print("--- SUCCESS: sign_language_model.p generated! ---")
else:
    print("CRITICAL ERROR: Path galat hai ya images nahi mili! Path check karo.")