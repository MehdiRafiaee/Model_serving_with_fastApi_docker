# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
import cv2
from app.model import predict, load_model
import io

app = FastAPI(title="Simple Model Serving API", version="1.0")

# وقتی سرویس بالا می‌آید، مدل را لود کن
@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Model Serving API!"}

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    # بررسی اینکه فایل تصویر است
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image!")
    
    # خواندن تصویر آپلود شده
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image!")
    
    # انجام پیش‌بینی
    try:
        prediction = predict(image)
        return {"filename": file.filename, "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# یک endpoint برای سلامت سرویس
@app.get("/health")
def health_check():
    return {"status": "healthy"}
