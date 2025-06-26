#!/usr/bin/env python3
"""
Test script for the MNIST Transformer API
Tests all endpoints and functionality
"""

import requests
import base64
# import json
import time
from PIL import Image, ImageDraw
# import numpy as np
import io

class APITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self):
        """Test health endpoint"""
        print("🔍 Testing health endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_root(self):
        """Test root endpoint"""
        print("🔍 Testing root endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Root endpoint: {data['message']}")
                return True
            else:
                print(f"❌ Root endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Root endpoint error: {e}")
            return False
    
    def create_test_image(self, digit=5, size=(28, 28)):
        """Create a simple test image with a digit"""
        # Create a white image
        image = Image.new('L', size, 255)
        draw = ImageDraw.Draw(image)
        
        # Draw a simple digit (just a rectangle for testing)
        margin = 4
        draw.rectangle([margin, margin, size[0]-margin, size[1]-margin], fill=0)
        
        return image
    
    def test_predict_file(self):
        """Test file upload prediction"""
        print("🔍 Testing file upload prediction...")
        try:
            # Create test image
            test_image = self.create_test_image()
            
            # Save to bytes
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Upload file
            files = {'file': ('test_digit.png', img_bytes, 'image/png')}
            response = self.session.post(f"{self.base_url}/predict", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ File prediction: digit={data['prediction']}, confidence={data['confidence']:.3f}")
                return True
            else:
                print(f"❌ File prediction failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ File prediction error: {e}")
            return False
    
    def test_predict_base64(self):
        """Test base64 prediction"""
        print("🔍 Testing base64 prediction...")
        try:
            # Create test image
            test_image = self.create_test_image()
            
            # Convert to base64
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='PNG')
            img_str = base64.b64encode(img_bytes.getvalue()).decode()
            
            # Send request
            payload = {"image": img_str}
            response = self.session.post(
                f"{self.base_url}/predict_base64",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Base64 prediction: digit={data['prediction']}, confidence={data['confidence']:.3f}")
                return True
            else:
                print(f"❌ Base64 prediction failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Base64 prediction error: {e}")
            return False
    
    def test_predict_batch(self):
        """Test batch prediction"""
        print("🔍 Testing batch prediction...")
        try:
            # Create multiple test images
            files = []
            for i in range(3):
                test_image = self.create_test_image(digit=i)
                img_bytes = io.BytesIO()
                test_image.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                files.append(('files', (f'test_digit_{i}.png', img_bytes, 'image/png')))
            
            # Send batch request
            response = self.session.post(f"{self.base_url}/predict_batch", files=files)
            
            if response.status_code == 200:
                data = response.json()
                results = data['results']
                print(f"✅ Batch prediction: {len(results)} images processed")
                for result in results:
                    print(f"   - {result['filename']}: digit={result['prediction']}, confidence={result['confidence']:.3f}")
                return True
            else:
                print(f"❌ Batch prediction failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Batch prediction error: {e}")
            return False
    
    def test_error_handling(self):
        """Test error handling"""
        print("🔍 Testing error handling...")
        
        # Test invalid file type
        try:
            files = {'file': ('test.txt', b'not an image', 'text/plain')}
            response = self.session.post(f"{self.base_url}/predict", files=files)
            if response.status_code == 400:
                print("✅ Invalid file type handled correctly")
            else:
                print(f"❌ Invalid file type not handled: {response.status_code}")
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
        
        # Test missing image in base64
        try:
            payload = {"wrong_field": "data"}
            response = self.session.post(f"{self.base_url}/predict_base64", json=payload)
            if response.status_code == 400:
                print("✅ Missing image field handled correctly")
            else:
                print(f"❌ Missing image field not handled: {response.status_code}")
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🧪 Starting API Tests")
        print("=" * 50)
        
        tests = [
            ("Health Check", self.test_health),
            ("Root Endpoint", self.test_root),
            ("File Prediction", self.test_predict_file),
            ("Base64 Prediction", self.test_predict_base64),
            ("Batch Prediction", self.test_predict_batch),
            ("Error Handling", self.test_error_handling),
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n📋 {test_name}")
            print("-" * 30)
            try:
                success = test_func()
                results.append((test_name, success))
            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Summary")
        print("=" * 50)
        
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        for test_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {test_name}")
        
        print(f"\n🎯 Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! API is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the API server.")
        
        return passed == total

def main():
    """Main test function"""
    import sys
    
    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print(f"🚀 Testing API at: {base_url}")
    
    # Wait a bit for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    # Run tests
    tester = APITester(base_url)
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 