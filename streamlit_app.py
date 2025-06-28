import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
# import io
from pathlib import Path
from streamlit_drawable_canvas import st_canvas
# import matplotlib.pyplot as plt

from model_utils import MNISTTransformerModel, download_and_load_weights

# --- SequenceDecoder and CTC decode ---
class SequenceDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, logits_seq):
        out, _ = self.rnn(logits_seq)
        out = self.fc(out)
        return out

def ctc_greedy_decode(log_probs, blank=10):
    pred = torch.argmax(log_probs, dim=2).squeeze(1).tolist()
    decoded = []
    prev = None
    for p in pred:
        if p != blank and p != prev:
            decoded.append(p)
        prev = p
    return decoded

# --- Model loading ---
@st.cache_resource
def load_models():
    encoder = MNISTTransformerModel()
    decoder = SequenceDecoder(input_size=128, hidden_size=128, num_classes=11)
    # dec_path = Path(".tf_model_mnist/trained_model3/pytorch_dec_model_part3.bin")
    # if not dec_path.exists():
    #     st.error(f"Decoder weights not found at {dec_path}")
    #     st.stop()
    # decoder.load_state_dict(torch.load(dec_path, map_location="cpu", weights_only=True))
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

# --- Sliding window encoding ---
def encode_patch(one_image, model):
    img_embedded = model.preprocess_model(one_image)
    transformer_output = model.tf_model(img_embedded)
    cls_output = transformer_output[:, 0, :]
    logits = model.lin_class_m(cls_output)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    confidence = probs[0, pred].item()
    return logits, pred, confidence, cls_output

def slide_seq_img_encode(seq_img, model, slide_size=8):
    w_size = seq_img.shape[-1]
    patch_size = seq_img.shape[0]
    window_len = int((w_size - patch_size) / slide_size)
    pred_per_digit_logit = []
    pred_per_digit_pred = []
    pred_per_digit_conf = []
    pred_per_digit_cls_out = []
    for j in range(window_len):
        x_position = j * slide_size
        crop_img = seq_img[:, x_position:x_position+patch_size].unsqueeze(0).unsqueeze(0)
        logits, pred, confidence, cls_output = encode_patch(crop_img, model)
        pred_per_digit_logit.append(logits)
        pred_per_digit_pred.append(pred)
        pred_per_digit_conf.append(confidence)
        pred_per_digit_cls_out.append(cls_output)
    pred_per_digit_logit = torch.stack(pred_per_digit_logit).squeeze(1)
    pred_per_digit_cls_out = torch.stack(pred_per_digit_cls_out).squeeze(1)
    return pred_per_digit_logit, pred_per_digit_pred, pred_per_digit_conf, pred_per_digit_cls_out

def add_blank_class_logits(logits_10, threshold=0.5, blank_boost=5.0):
    probs = F.softmax(logits_10, dim=1)
    max_conf, _ = probs.max(dim=1)
    blank_logits = torch.full((logits_10.size(0), 1), fill_value=-10.0, device=logits_10.device)
    high_blank_indices = max_conf < threshold
    blank_logits[high_blank_indices] = blank_boost
    logits_11 = torch.cat([logits_10, blank_logits], dim=1)
    return logits_11

# --- Streamlit UI ---
st.set_page_config(page_title="Sequence Digit Recognizer (CTC)", page_icon="🔢", layout="wide")
st.title("🔢 Sequence Digit Recognizer (CTC)")
st.markdown("""
Upload or draw a sequence of handwritten digits (height=28, width=112). The model will predict the digit sequence using a transformer encoder and CTC decoder.
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
                    seq_img = np.array(image_28x140)
                    seq_img = ImageOps.invert(Image.fromarray(seq_img)).convert("L")
                    seq_img = np.array(seq_img) / 255.0
                    seq_img = torch.tensor(seq_img, dtype=torch.float32).to(device)
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
    st.header("Upload Sequence Image (28x112)")
    uploaded_file = st.file_uploader(
        "Choose an image file (28x112)",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image with a sequence of 4 handwritten digits (width=112, height=28)"
    )
    if uploaded_file is not None:
        # image = Image.open(uploaded_file).convert("L")
        image_28x112 = uploaded_file.resize((112, 28), Image.Resampling.LANCZOS)
        st.image(image_28x112, caption="Resized for model (28x112)", width=448)
        if st.button("🔍 Predict Sequence", key="upload_predict", type="primary"):
            with st.spinner("Predicting..."):
                result = None
                try:
                    # seq_img = np.array(image_28x112)
                    # seq_img = ImageOps.invert(Image.fromarray(seq_img)).convert("L")
                    # seq_img = np.array(seq_img) / 255.0
                    # seq_img = torch.tensor(seq_img, dtype=torch.float32).to(device)
                    # seq_img = seq_img.unsqueeze(0)
                    # seq_img = seq_img.squeeze(0)
                    _, _, pred_conf_10, pred_cls_out_10 = slide_seq_img_encode(image_28x112, encoder, slide_size=8)
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