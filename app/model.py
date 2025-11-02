# app/model.py
import joblib
import numpy as np
import cv2
from skimage import feature

model = None

def load_model():
    global model
    model_path = "app/models/model_weights.pkl"
    model = joblib.load(model_path)
    print("Model loaded successfully!")

def extract_hog_features(image):
    image = cv2.resize(image, (64, 128))
    hog_features = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), transform_sqrt=True, 
                              block_norm="L2", visualize=False)
    return hog_features

def predict(image):
    global model
    if model is None:
        raise RuntimeError("Model is not loaded!")
    
    features = extract_hog_features(image)
    features = features.reshape(1, -1)
    
    prediction = model.predict(features)
    return "Face" if prediction[0] == 1 else "Not Face"
