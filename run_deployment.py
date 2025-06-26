#!/usr/bin/env python3
"""
Deployment script for MNIST Transformer Model
Runs both FastAPI and Streamlit servers together
"""

import subprocess
import time
import sys
# import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'torchvision', 'fastapi', 'uvicorn', 
        'streamlit', 'PIL', 'numpy', 'matplotlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_model_files():
    """Check if trained model files exist"""
    model_path = Path(".tf_model_mnist/trained_model/")
    required_files = [
        "pytorch_pp_model.bin",
        "pytorch_tf_model.bin", 
        "pytorch_class_lin_model.bin"
    ]
    
    missing_files = []
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing model files: {', '.join(missing_files)}")
        print("The app will run with untrained models.")
        return False
    
    print("✅ All model files found")
    return True

def run_fastapi():
    """Run FastAPI server"""
    print("🚀 Starting FastAPI server...")
    try:
        # Run FastAPI server
        process = subprocess.Popen([
            sys.executable, "fastapi_app.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for server to start
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ FastAPI server started successfully")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start FastAPI server: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting FastAPI server: {e}")
        return None

def run_streamlit():
    """Run Streamlit app"""
    print("🎨 Starting Streamlit app...")
    try:
        # Run Streamlit
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for app to start
        time.sleep(5)
        
        if process.poll() is None:
            print("✅ Streamlit app started successfully")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start Streamlit app: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting Streamlit app: {e}")
        return None

def main():
    """Main deployment function"""
    print("🔢 MNIST Transformer Deployment")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check model files
    check_model_files()
    
    print("\n📋 Starting services...")
    
    # Start FastAPI server
    fastapi_process = run_fastapi()
    if not fastapi_process:
        print("❌ Cannot start deployment without FastAPI server")
        return
    
    # Start Streamlit app
    streamlit_process = run_streamlit()
    if not streamlit_process:
        print("❌ Cannot start deployment without Streamlit app")
        fastapi_process.terminate()
        return
    
    print("\n🎉 Deployment successful!")
    print("=" * 40)
    print("📊 FastAPI Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🎨 Streamlit App: http://localhost:8501")
    print("\nPress Ctrl+C to stop all services")
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if fastapi_process.poll() is not None:
                print("❌ FastAPI server stopped unexpectedly")
                break
                
            if streamlit_process.poll() is not None:
                print("❌ Streamlit app stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        
        # Terminate processes
        if fastapi_process:
            fastapi_process.terminate()
            print("✅ FastAPI server stopped")
            
        if streamlit_process:
            streamlit_process.terminate()
            print("✅ Streamlit app stopped")
            
        print("👋 Deployment stopped")

if __name__ == "__main__":
    main() 