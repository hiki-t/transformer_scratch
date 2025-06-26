import streamlit as st
import numpy as np
import requests
# import json
import base64
from PIL import Image # , ImageDraw
import io
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# this is for only using streamli app
from model_utils import MNISTTransformerModel as mnist_model

# Page configuration
st.set_page_config(
    page_title="MNIST Transformer Classifier",
    page_icon="🔢",
    layout="wide"
)

# Title and description
st.title("🔢 MNIST Digit Classifier")
st.markdown("""
This app uses a Vision Transformer model to classify handwritten digits (0-9).
You can either draw a digit or upload an image to get predictions.
""")

### this is for local setting
# # Sidebar for API configuration
# st.sidebar.header("API Configuration")
# api_url = st.sidebar.text_input(
#     "API URL", 
#     value="http://localhost:8000",
#     help="URL of the FastAPI server"
# )

### this is for local setting
# # Check API health
# @st.cache_data(ttl=60)
# def check_api_health(url):
#     """Check if the API is running"""
#     try:
#         response = requests.get(f"{url}/health", timeout=5)
#         return response.status_code == 200, response.json()
#     except:
#         return False, None

### this is for local setting
# # Health check
# is_healthy, health_data = check_api_health(api_url)

# if not is_healthy:
#     st.error(f"❌ Cannot connect to API at {api_url}")
#     st.info("Make sure the FastAPI server is running with: `python fastapi_app.py`")
#     st.stop()

### this is for local setting
# st.success(f"✅ Connected to API at {api_url}")
# if health_data:
#     st.sidebar.json(health_data)

# Main content
tab1, tab2, tab3 = st.tabs(["🎨 Draw Digit", "📁 Upload Image", "📊 Batch Upload"])

with tab1:
    st.header("Draw a Digit")
    st.markdown("Draw a digit (0-9) in the canvas below:")
    
    # Create drawing canvas
    canvas_result = st_canvas(
        stroke_width=20,
        stroke_color="#000000",
        background_color="#FFFFFF",
        update_streamlit=True,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        # Convert canvas to image and resize to 28x28
        image = Image.fromarray(canvas_result.image_data)
        
        # Resize to 28x28 for the model
        image_28x28 = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Display the drawn image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Your Drawing")
            st.image(image, width=200)
            st.caption("Original canvas (280x280)")
            
            # Also show the resized version
            st.image(image_28x28, width=100)
            st.caption("Resized for model (28x28)")
        
        with col2:
            st.subheader("Prediction")
            
            if st.button("🔍 Predict Digit", type="primary"):
                with st.spinner("Making prediction..."):
                    try:
                        # Convert resized image to base64
                        buffered = io.BytesIO()
                        image_28x28.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        
                        # # Send to API
                        # response = requests.post(
                        #     f"{api_url}/predict_base64",
                        #     json={"image": img_str},
                        #     timeout=30
                        # )
                        
                        # if response.status_code == 200:
                        #     result = response.json()
                            
                        #     # Display prediction
                        #     st.metric(
                        #         label="Predicted Digit", 
                        #         value=result["prediction"],
                        #         delta=f"{result['confidence']:.1%} confidence"
                        #     )
                            
                        #     # Display confidence bar
                        #     st.progress(result["confidence"])
                            
                        #     # Display all probabilities
                        #     st.subheader("All Probabilities")
                        #     probs = result["probabilities"]
                        #     digits = list(range(10))
                            
                        #     # Create bar chart
                        #     fig, ax = plt.subplots(figsize=(10, 4))
                        #     bars = ax.bar(digits, probs, color='skyblue', alpha=0.7)
                        #     ax.set_xlabel('Digit')
                        #     ax.set_ylabel('Probability')
                        #     ax.set_title('Prediction Probabilities')
                        #     ax.set_xticks(digits)
                            
                        #     # Highlight the predicted digit
                        #     bars[result["prediction"]].set_color('red')
                            
                        #     st.pyplot(fig)
                            
                        # else:
                        #     st.error(f"API Error: {response.text}")

                        # Direct call to your model class
                        result = mnist_model.predict(image_28x28)  # or whatever your processed image variable is called

                        # Display prediction
                        st.metric(
                            label="Predicted Digit", 
                            value=result["prediction"],
                            delta=f"{result['confidence']:.1%} confidence"
                        )

                        # Display confidence bar
                        st.progress(result["confidence"])

                        # Display all probabilities
                        st.subheader("All Probabilities")
                        probs = result["probabilities"]
                        digits = list(range(10))

                        # Create bar chart
                        fig, ax = plt.subplots(figsize=(10, 4))
                        bars = ax.bar(digits, probs, color='skyblue', alpha=0.7)
                        ax.set_xlabel('Digit')
                        ax.set_ylabel('Probability')
                        ax.set_title('Prediction Probabilities')
                        ax.set_xticks(digits)

                        # Highlight the predicted digit
                        bars[result["prediction"]].set_color('red')

                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")

with tab2:
    st.header("Upload Image")
    st.markdown("Upload an image containing a handwritten digit:")
    
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image with a handwritten digit (0-9)"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, width=200)
        
        with col2:
            st.subheader("Prediction")
            
            if st.button("🔍 Predict Digit", key="upload_predict", type="primary"):
                with st.spinner("Making prediction..."):
                    try:
                        # Prepare file for upload
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        
                        # # Send to API
                        # response = requests.post(
                        #     f"{api_url}/predict",
                        #     files=files,
                        #     timeout=30
                        # )
                        
                        # if response.status_code == 200:
                        #     result = response.json()
                            
                        #     # Display prediction
                        #     st.metric(
                        #         label="Predicted Digit", 
                        #         value=result["prediction"],
                        #         delta=f"{result['confidence']:.1%} confidence"
                        #     )
                            
                        #     # Display confidence bar
                        #     st.progress(result["confidence"])
                            
                        #     # Display all probabilities
                        #     st.subheader("All Probabilities")
                        #     probs = result["probabilities"]
                        #     digits = list(range(10))
                            
                        #     # Create bar chart
                        #     fig, ax = plt.subplots(figsize=(10, 4))
                        #     bars = ax.bar(digits, probs, color='lightgreen', alpha=0.7)
                        #     ax.set_xlabel('Digit')
                        #     ax.set_ylabel('Probability')
                        #     ax.set_title('Prediction Probabilities')
                        #     ax.set_xticks(digits)
                            
                        #     # Highlight the predicted digit
                        #     bars[result["prediction"]].set_color('red')
                            
                        #     st.pyplot(fig)
                            
                        # else:
                        #     st.error(f"API Error: {response.text}")

                        image = Image.open(io.BytesIO(uploaded_file.getvalue()))

                        # Direct call to your model class
                        result = mnist_model.predict(image)

                        # Display prediction
                        st.metric(
                            label="Predicted Digit", 
                            value=result["prediction"],
                            delta=f"{result['confidence']:.1%} confidence"
                        )

                        # Display confidence bar
                        st.progress(result["confidence"])

                        # Display all probabilities
                        st.subheader("All Probabilities")
                        probs = result["probabilities"]
                        digits = list(range(10))

                        # Create bar chart
                        fig, ax = plt.subplots(figsize=(10, 4))
                        bars = ax.bar(digits, probs, color='lightgreen', alpha=0.7)
                        ax.set_xlabel('Digit')
                        ax.set_ylabel('Probability')
                        ax.set_title('Prediction Probabilities')
                        ax.set_xticks(digits)

                        # Highlight the predicted digit
                        bars[result["prediction"]].set_color('red')

                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")

with tab3:
    st.header("Batch Upload")
    st.markdown("Upload multiple images to classify them all at once:")
    
    uploaded_files = st.file_uploader(
        "Choose multiple image files", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        accept_multiple_files=True,
        help="Upload multiple images with handwritten digits"
    )
    
    if uploaded_files:
        st.subheader(f"Uploaded {len(uploaded_files)} images")
        
        # Display thumbnails
        cols = st.columns(min(4, len(uploaded_files)))
        for i, uploaded_file in enumerate(uploaded_files):
            with cols[i % 4]:
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, width=100)
        
        if st.button("🔍 Predict All Digits", type="primary"):
            with st.spinner("Making batch predictions..."):
                try:
                    # # Prepare files for upload
                    # files = []
                    # for uploaded_file in uploaded_files:
                    #     files.append(("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)))
                    
                    # # Send to API
                    # response = requests.post(
                    #     f"{api_url}/predict_batch",
                    #     files=files,
                    #     timeout=60
                    # )
                    
                    # if response.status_code == 200:
                    #     results = response.json()["results"]
                        
                    #     st.subheader("Batch Results")
                        
                    #     # Create results table
                    #     results_data = []
                    #     for result in results:
                    #         results_data.append({
                    #             "Filename": result["filename"],
                    #             "Prediction": result["prediction"],
                    #             "Confidence": f"{result['confidence']:.1%}"
                    #         })
                        
                    #     st.dataframe(results_data, use_container_width=True)
                        
                    #     # Summary statistics
                    #     predictions = [r["prediction"] for r in results]
                    #     confidences = [r["confidence"] for r in results]
                        
                    #     col1, col2, col3 = st.columns(3)
                    #     with col1:
                    #         st.metric("Total Images", len(results))
                    #     with col2:
                    #         st.metric("Avg Confidence", f"{np.mean(confidences):.1%}")
                    #     with col3:
                    #         st.metric("Most Common", max(set(predictions), key=predictions.count))
                        
                    # else:
                    #     st.error(f"API Error: {response.text}")

                    results = []
                    for uploaded_file in uploaded_files:
                        # Read each uploaded file as a PIL image
                        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
                        # Direct call to your model class
                        result = mnist_model.predict(image)
                        results.append({
                            "filename": uploaded_file.name,
                            "prediction": result["prediction"],
                            "confidence": result["confidence"]
                        })

                    st.subheader("Batch Results")

                    # Create results table
                    results_data = []
                    for result in results:
                        results_data.append({
                            "Filename": result["filename"],
                            "Prediction": result["prediction"],
                            "Confidence": f"{result['confidence']:.1%}"
                        })

                    st.dataframe(results_data, use_container_width=True)

                    # Summary statistics
                    predictions = [r["prediction"] for r in results]
                    confidences = [r["confidence"] for r in results]

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Images", len(results))
                    with col2:
                        st.metric("Avg Confidence", f"{np.mean(confidences):.1%}")
                    with col3:
                        st.metric("Most Common", max(set(predictions), key=predictions.count))

                except Exception as e:
                    st.error(f"Error making batch predictions: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit and FastAPI | Vision Transformer for MNIST Classification</p>
</div>
""", unsafe_allow_html=True) 