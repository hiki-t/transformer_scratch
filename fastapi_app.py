from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
import uvicorn
import io
import base64
from PIL import Image
# import numpy as np
# import json
from model_utils import MNISTTransformerModel

# Initialize FastAPI app
app = FastAPI(
    title="MNIST Transformer API",
    description="API for digit classification using Vision Transformer",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model (global variable)
model = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global model
    try:
        model = MNISTTransformerModel()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MNIST Transformer API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict digit from uploaded image",
            "/predict_base64": "POST - Predict digit from base64 encoded image",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(model.device) if model else None
    }

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    """Predict digit from uploaded image file"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Make prediction
        result = model.predict(image)
        
        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "filename": file.filename
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_base64")
async def predict_digit_base64(data: dict):
    """Predict digit from base64 encoded image"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Extract base64 data
        if "image" not in data:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request")
        
        # Decode base64 image
        image_data = base64.b64decode(data["image"])
        
        image = Image.open(io.BytesIO(image_data))
        
        # Make prediction
        result = model.predict(image)
        
        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Predict digits from multiple uploaded images"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        results = []
        for file in files:
            # Validate file type
            if not file.content_type.startswith("image/"):
                continue
            
            # Read and process image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            
            # Make prediction
            result = model.predict(image)
            results.append({
                "filename": file.filename,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"]
            })
        
        return {"results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 