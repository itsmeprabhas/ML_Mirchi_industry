import io
import time
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Directories
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "exports" / "mirchi_grader.onnx"

# Config
CLASSES = ["Grade_A", "Grade_B", "Grade_C", "Reject"]
IMAGE_SIZE = 224

app = FastAPI(title="ML Mirchi API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ONNX model at startup
session = None
if MODEL_PATH.exists():
    session = ort.InferenceSession(str(MODEL_PATH))
else:
    print(f"Warning: Model not found at {MODEL_PATH}. Make sure to run export_model.py first.")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess the image to match the training pipeline."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # HWC to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension
    return np.expand_dims(img_array, axis=0)

def softmax(x):
    """Compute softmax values for logits."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if session is None:
        return {"error": "Model not loaded"}

    try:
        contents = await file.read()
        
        start_time = time.perf_counter()
        
        # Preprocess
        input_data = preprocess_image(contents)
        
        # Inference
        input_name = session.get_inputs()[0].name
        logits = session.run(None, {input_name: input_data})[0]
        
        # Postprocess
        probs = softmax(logits)[0]
        pred_idx = np.argmax(probs)
        pred_class = CLASSES[pred_idx]
        confidence = probs[pred_idx]
        
        latency = (time.perf_counter() - start_time) * 1000  # ms
        
        # Map probabilities back to classes for the frontend
        class_probs = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
        
        return {
            "grade": pred_class,
            "confidence": float(confidence),
            "latency_ms": float(latency),
            "probabilities": class_probs
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": session is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
