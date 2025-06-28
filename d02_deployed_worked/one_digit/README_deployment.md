# MNIST Transformer Deployment

This project provides a complete deployment solution for the MNIST Vision Transformer model using FastAPI and Streamlit.

## 🏗️ Architecture

```
┌─────────────────┐    HTTP Requests    ┌─────────────────┐
│   Streamlit     │ ──────────────────► │   FastAPI       │
│   Frontend      │                     │   Backend       │
│   (Port 8501)   │ ◄────────────────── │   (Port 8000)   │
└─────────────────┘    JSON Responses   └─────────────────┘
                                                │
                                                ▼
                                        ┌─────────────────┐
                                        │   PyTorch       │
                                        │   Model         │
                                        └─────────────────┘
```

## 📁 Project Structure

```
transformer_scratch/
├── tm01_toy_model_run1.py      # Original training script
├── model_utils.py              # Model classes and prediction logic
├── fastapi_app.py              # FastAPI backend server
├── streamlit_app.py            # Streamlit frontend application
├── run_deployment.py           # Deployment orchestration script
├── requirements.txt            # Python dependencies
├── README_deployment.md        # This file
└── tf_model_mnist/
    └── trained_model/          # Trained model weights
        ├── pytorch_pp_model.bin
        ├── pytorch_tf_model.bin
        └── pytorch_class_lin_model.bin
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)

If you don't have trained model weights, run the training script:

```bash
python tm01_toy_model_run1.py
```

This will:

- Download MNIST dataset
- Train the Vision Transformer model
- Save model weights to `tf_model_mnist/trained_model/`
- Upload models to Hugging Face Hub

### 3. Deploy the Application

#### Option A: Run Everything Together

```bash
python run_deployment.py
```

#### Option B: Run Services Separately

**Terminal 1 - Start FastAPI Backend:**

```bash
python fastapi_app.py
```

**Terminal 2 - Start Streamlit Frontend:**

```bash
streamlit run streamlit_app.py
```

### 4. Access the Application

- **Streamlit App**: http://localhost:8501
- **FastAPI Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🎨 Features

### Streamlit Frontend

- **Interactive Drawing Canvas**: Draw digits with your mouse
- **File Upload**: Upload image files for classification
- **Batch Processing**: Upload multiple images at once
- **Real-time Predictions**: See confidence scores and probability distributions
- **Beautiful Visualizations**: Bar charts showing prediction probabilities

### FastAPI Backend

- **RESTful API**: Clean, documented endpoints
- **Multiple Input Formats**: File upload and base64 encoding
- **Batch Processing**: Handle multiple images efficiently
- **Health Checks**: Monitor service status
- **CORS Support**: Cross-origin requests enabled
- **Auto-generated Docs**: Interactive API documentation

## 📡 API Endpoints

### Health Check

```http
GET /health
```

### Single Image Prediction

```http
POST /predict
Content-Type: multipart/form-data
```

### Base64 Image Prediction

```http
POST /predict_base64
Content-Type: application/json
{
    "image": "base64_encoded_image_string"
}
```

### Batch Prediction

```http
POST /predict_batch
Content-Type: multipart/form-data
```

## 🔧 Configuration

### Model Configuration

The model configuration is defined in `model_utils.py`:

```python
config = {
    "patch_size": 7,        # Size of image patches
    "patch_len": 16,        # Number of patches (28/7 = 4, 4² = 16)
    "model_d": 128,         # Model dimension
    "num_layers": 2,        # Number of transformer layers
    "dropout": 0.1,         # Dropout rate
    "num_heads": 2,         # Number of attention heads
}
```

### Server Configuration

- **FastAPI**: Port 8000 (configurable in `fastapi_app.py`)
- **Streamlit**: Port 8501 (configurable in `run_deployment.py`)

## 🐳 Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["python", "run_deployment.py"]
```

Build and run:

```bash
docker build -t mnist-transformer .
docker run -p 8000:8000 -p 8501:8501 mnist-transformer
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**

   ```bash
   pip install -r requirements.txt
   ```

2. **Model Not Found**

   - Ensure trained model files exist in `tf_model_mnist/trained_model/`
   - Run training script if needed: `python tm01_toy_model_run1.py`

3. **Port Already in Use**

   - Change ports in respective files
   - Kill existing processes: `lsof -ti:8000 | xargs kill -9`

4. **CUDA/GPU Issues**
   - The model automatically uses CPU if CUDA is not available
   - Check device in health endpoint: `GET /health`

### Debug Mode

Run with verbose output:

```bash
python -u run_deployment.py
```

## 📊 Performance

- **Inference Time**: ~50-100ms per image (CPU)
- **Memory Usage**: ~200MB (model + dependencies)
- **Concurrent Requests**: FastAPI handles multiple requests efficiently

## 🔒 Security Considerations

- **Input Validation**: All image inputs are validated
- **File Size Limits**: Configure in FastAPI settings
- **CORS**: Configure allowed origins for production
- **Rate Limiting**: Consider adding rate limiting for production

## 🚀 Production Deployment

For production deployment:

1. **Use a production WSGI server** (Gunicorn + Uvicorn)
2. **Add reverse proxy** (Nginx)
3. **Implement authentication** if needed
4. **Add monitoring and logging**
5. **Use environment variables** for configuration
6. **Set up SSL/TLS certificates**

Example production command:

```bash
gunicorn fastapi_app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📝 License

This project is part of the MLX8 course materials.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!
