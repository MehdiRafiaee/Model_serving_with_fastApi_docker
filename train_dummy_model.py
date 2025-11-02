# train_dummy_model.py
import cv2
import numpy as np
import joblib
from skimage import feature
from skimage import exposure
import os

# تابع برای استخراج ویژگی‌های HOG از تصویر
def extract_hog_features(image_path):
    image = cv2.imread(image_path, 0)  # خواندن تصویر به صورت خاکستری
    if image is None:
        return None
    image = cv2.resize(image, (64, 128))  # تغییر سایز به ابعاد ثابت
    
    # محاسبه HOG
    (hog_features, hog_image) = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                                            cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2",
                                            visualize=True)
    return hog_features

# ساخت داده‌های مصنوعی و آموزشی ساده (غیرواقعی)
def create_dummy_data():
    features = []
    labels = []
    
    # فرض کنید دو کلاس "چهره" و "غیر چهره" داریم
    for i in range(20):
        hog_feat = np.random.rand(3780)  # این عدد بُعد HOG برای ابعاد 64x128 است
        features.append(hog_feat)
        labels.append(i % 2)  # برچسب‌های 0 و 1
        
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    # ساخت داده‌های مصنوعی
    X, y = create_dummy_data()
    
    # آموزش یک مدل ساده
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score
    
    model = LinearSVC(random_state=42)
    model.fit(X, y)
    
    # ذخیره مدل
    joblib.dump(model, "app/models/model_weights.pkl")
    print("Model trained and saved!")
