#!/bin/bash

# MNIST Transformer Deployment Startup Script

echo "🔢 MNIST Transformer Deployment"
echo "================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "tm01_toy_model_run1.py" ]; then
    echo "❌ Please run this script from the transformer_scratch directory"
    exit 1
fi

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
    echo "✅ Dependencies installed"
fi

# # Check if model files exist
# if [ ! -d ".tf_model_mnist/trained_model" ]; then
#     echo "⚠️  No trained model found. Would you like to train the model first? (y/n)"
#     read -r response
#     if [[ "$response" =~ ^[Yy]$ ]]; then
#         echo "🚀 Training model..."
#         python tm01_toy_model_run1.py
#         if [ $? -ne 0 ]; then
#             echo "❌ Training failed"
#             exit 1
#         fi
#         echo "✅ Model training completed"
#     else
#         echo "⚠️  Running with untrained model"
#     fi
# fi

# Start the deployment
echo "🚀 Starting deployment..."
python run_deployment.py

echo "👋 Deployment stopped" 