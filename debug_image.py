#!/usr/bin/env python3
"""
Debug script to test image processing and identify shape issues
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
from model_utils import MNISTTransformerModel

def create_test_canvas_image():
    """Create a test image similar to what the canvas produces"""
    # Create a 280x280 RGBA image (like canvas)
    image = Image.new('RGBA', (280, 280), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # Draw a simple digit (rectangle)
    margin = 40
    draw.rectangle([margin, margin, 280-margin, 280-margin], fill=(0, 0, 0, 255))
    
    return image

def test_image_processing():
    """Test the image processing pipeline"""
    print("🔍 Testing image processing pipeline...")
    
    # Create test image
    test_image = create_test_canvas_image()
    print(f"Original image size: {test_image.size}, mode: {test_image.mode}")
    
    # Convert to numpy array (like canvas does)
    image_array = np.array(test_image)
    print(f"Array shape: {image_array.shape}, dtype: {image_array.dtype}")
    
    # Initialize model
    model = MNISTTransformerModel()
    
    # Test preprocessing
    try:
        img_tensor = model.preprocess_image(image_array)
        print(f"Preprocessed tensor shape: {img_tensor.shape}")
        
        # Test patch embedding
        img_embedded = model.preprocess_model(img_tensor)
        print(f"Embedded tensor shape: {img_embedded.shape}")
        
        # Test transformer
        transformer_output = model.tf_model(img_embedded)
        print(f"Transformer output shape: {transformer_output.shape}")
        
        # Test classification
        cls_output = transformer_output[:, 0, :]
        print(f"CLS token shape: {cls_output.shape}")
        
        logits = model.lin_class_m(cls_output)
        print(f"Logits shape: {logits.shape}")
        
        # Get prediction
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0, prediction].item()
        
        print(f"✅ Success! Prediction: {prediction}, Confidence: {confidence:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_base64_encoding():
    """Test base64 encoding/decoding"""
    print("\n🔍 Testing base64 encoding...")
    
    # Create test image
    test_image = create_test_canvas_image()
    
    # Resize to 28x28
    test_image_28x28 = test_image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to base64
    buffered = io.BytesIO()
    test_image_28x28.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    print(f"Base64 string length: {len(img_str)}")
    
    # Decode back
    image_data = base64.b64decode(img_str)
    decoded_image = Image.open(io.BytesIO(image_data))
    
    print(f"Decoded image size: {decoded_image.size}, mode: {decoded_image.mode}")
    
    return img_str

def test_api_call():
    """Test API call with base64 image"""
    print("\n🔍 Testing API call...")
    
    import requests
    
    # Create base64 image
    img_str = test_base64_encoding()
    
    # Test API call
    try:
        response = requests.post(
            "http://localhost:8000/predict_base64",
            json={"image": img_str},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API call successful: {result}")
            return True
        else:
            print(f"❌ API call failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ API call error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 MNIST Transformer Debug Script")
    print("=" * 40)
    
    # Test image processing
    success1 = test_image_processing()
    
    # Test base64 encoding
    test_base64_encoding()
    
    # Test API call (only if server is running)
    try:
        success2 = test_api_call()
    except:
        print("⚠️  API server not running, skipping API test")
        success2 = True
    
    print("\n" + "=" * 40)
    if success1 and success2:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above.") 