import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from model_utils import MNISTTransformerModel, download_and_load_weights, SequenceDecoder, ctc_greedy_decode, slide_seq_img_encode

# --- Model loading ---
@st.cache_resource
def load_models():
    encoder = MNISTTransformerModel()
    decoder = SequenceDecoder(input_size=128, hidden_size=128, num_classes=11)
    decoder = download_and_load_weights(decoder, "pytorch_dec_model_part3.bin")
    encoder.preprocess_model.eval()
    encoder.tf_model.eval()
    encoder.lin_class_m.eval()
    decoder.eval()
    return encoder, decoder

encoder, decoder = load_models()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.preprocess_model.to(device)
encoder.tf_model.to(device)
encoder.lin_class_m.to(device)
decoder.to(device)

# --- Streamlit UI ---
st.set_page_config(page_title="Sequence Digit Recognizer (CTC)", page_icon="🔢", layout="wide")
st.title("🔢 Sequence Digit Recognizer (CTC)")
st.markdown("""
Upload or draw a sequence of handwritten digits (height=28, width=140). The model will predict the digit sequence using a transformer encoder and CTC decoder.
""")

tab1, tab2 = st.tabs(["🎨 Draw Sequence", "📁 Upload Image"])

with tab1:
    st.header("Draw a Sequence (4 digits)")
    st.markdown("Draw a sequence of 4 digits (left to right) in the canvas below:")
    canvas_result = st_canvas(
        stroke_width=20,
        stroke_color="#000000",
        background_color="#FFFFFF",
        update_streamlit=True,
        height=112,
        width=560,
        drawing_mode="freedraw",
        key="canvas_seq",
    )
    if canvas_result.image_data is not None:
        image = Image.fromarray(canvas_result.image_data)
        image_28x140 = image.resize((140, 28), Image.Resampling.LANCZOS).convert("L")
        st.image(image_28x140, caption="Resized for model (28x140)", width=560)
        if st.button("🔍 Predict Sequence", key="draw_predict", type="primary"):
            with st.spinner("Predicting..."):
                result = None
                try:
                    # ... after image is grayscale and 28x28
                    image_28x140 = ImageOps.invert(image_28x140)
                    seq_img = np.array(image_28x140)
                    # seq_img = ImageOps.invert(Image.fromarray(seq_img)).convert("L")
                    seq_img = np.array(seq_img) / 255.0
                    seq_img = encoder.transform(seq_img)
                    seq_img = torch.tensor(seq_img, dtype=torch.float32).to(device)
                    print(seq_img.shape)
                    # seq_img = seq_img.unsqueeze(0)  # [1, 28, 140]
                    # seq_img = seq_img.squeeze(0)    # [28, 140]

                    # Sliding window encoding
                    _, _, pred_conf_10, pred_cls_out_10 = slide_seq_img_encode(seq_img, encoder, slide_size=8)
                    conf_tensor = torch.tensor(pred_conf_10).unsqueeze(1).to(device)
                    weighted_cls = conf_tensor * pred_cls_out_10
                    decoder_output = decoder(weighted_cls.unsqueeze(0))  # [1, T, 11]
                    log_probs = F.log_softmax(decoder_output, dim=2).transpose(0, 1)  # [T, 1, C]
                    decoded = ctc_greedy_decode(log_probs, blank=10)
                    st.subheader(f"Predicted Sequence: {''.join(map(str, decoded))}")
                    st.write(f"Decoded digits: {decoded}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

with tab2:
    st.header("Upload Sequence Image (28x140)")
    uploaded_file = st.file_uploader(
        "Choose an image file (28x140)",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image with a sequence of 4 handwritten digits (width=140, height=28)"
    )
    if uploaded_file is not None:
        # image = Image.open(uploaded_file).convert("L")
        image_28x140 = uploaded_file.resize((140, 28), Image.Resampling.LANCZOS)
        st.image(image_28x140, caption="Resized for model (28x140)", width=448)
        if st.button("🔍 Predict Sequence", key="upload_predict", type="primary"):
            with st.spinner("Predicting..."):
                result = None
                try:
                    # seq_img = np.array(image_28x140)
                    # seq_img = ImageOps.invert(Image.fromarray(seq_img)).convert("L")
                    # seq_img = np.array(seq_img) / 255.0
                    # seq_img = torch.tensor(seq_img, dtype=torch.float32).to(device)
                    # seq_img = seq_img.unsqueeze(0)
                    # seq_img = seq_img.squeeze(0)
                    _, _, pred_conf_10, pred_cls_out_10 = slide_seq_img_encode(image_28x140, encoder, slide_size=8)
                    conf_tensor = torch.tensor(pred_conf_10).unsqueeze(1).to(device)
                    weighted_cls = conf_tensor * pred_cls_out_10
                    decoder_output = decoder(weighted_cls.unsqueeze(0))
                    log_probs = F.log_softmax(decoder_output, dim=2).transpose(0, 1)
                    decoded = ctc_greedy_decode(log_probs, blank=10)
                    st.subheader(f"Predicted Sequence: {''.join(map(str, decoded))}")
                    st.write(f"Decoded digits: {decoded}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit | Vision Transformer + CTC for Sequence Digit Recognition</p>
</div>
""", unsafe_allow_html=True) 