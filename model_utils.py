import torch
import torch.nn as nn
import math
from einops import rearrange
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
# import os
# from datetime import datetime

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model):
        super(SingleHeadAttention, self).__init__()
        
        # Initialize dimensions
        self.d_model = d_model # model's dim
        self.d_k = d_model
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Softmax to obtain attention prob
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
                
    def forward(self, x):
        # Apply linear transformations
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        
        # Perform scaled dot-product to get attention
        attn_output = self.scaled_dot_product_attention(Q, K, V)
        return self.W_o(attn_output)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        # Q, K, V: [batch, num_heads, seq_len, d_k]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V) # [batch, num_heads, seq_len, seq_len] * [batch, num_heads, seq_len, d_k]
        return output

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        # Linear projections
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads and reshape [batch, seq_len, num_head, d_k] -> [batch, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention on all heads
        attn_output = self.scaled_dot_product_attention(Q, K, V)  # [batch, num_heads, seq_len, d_k]

        # Concat heads, 1.transpose [..., num_heads, seq_len, ...] -> [..., seq_len, num_heads, d_k], 2. concat [num_heads, d_k] -> d_model
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # [batch, seq_len, d_model]

        # Final linear layer
        return self.W_o(attn_output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, patch_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(patch_length, d_model)
        position = torch.arange(0, patch_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0)) # pe shape [1, patch_length, d_model]
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)] # selects the positional encodings for the current sequence length.

class EncoderLayer(nn.Module):
    def __init__(self, model_d, dropout, num_heads=1, mlp_ratio=4):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(model_d, num_heads)
        self.norm1 = nn.LayerNorm(model_d)
        self.norm2 = nn.LayerNorm(model_d)
        self.dropout = nn.Dropout(dropout)
        # Feed-forward network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(model_d, model_d * mlp_ratio),
            nn.GELU(),  # or nn.ReLU()
            nn.Dropout(dropout),
            nn.Linear(model_d * mlp_ratio, model_d),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [batch, num_patches+1, model_d]
        # Self-attention + residual + norm
        attn_out = self.self_attn(x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feed-forward(lin or MLP) + residual + norm
        out = self.mlp(x)
        x = x + out
        x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, model_d, num_layers, dropout, num_heads=1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(model_d, dropout, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class PatchEmbedder(nn.Module):
    def __init__(self, patch_size, model_d, patch_len):
        super().__init__()
        self.patch_size = patch_size # h or w of each patch img
        self.pixel_len = patch_size * patch_size # h*w
        self.lin_emb = nn.Linear(self.pixel_len, model_d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_d)) # nn.Parameter(torch.zeros(1, 1, model_d))
        self.posi_enc = PositionalEncoding(model_d, patch_len+1) # PositionalEncoding(model_d, self.num_patches+1)
        
    def forward(self, x):
        # x: [batch, channels, height, width]
        img_t = rearrange(
            x, 
            'b c (h p1) (w p2) -> b (h w) (p1 p2) c', 
            p1=self.patch_size, p2=self.patch_size
        )
        img_t = img_t.squeeze(-1)  # [batch, num_patches, pixel_len]
        img_t = self.lin_emb(img_t)  # [batch, num_patches, model_d]
        batch_size = img_t.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, model_d]
        img_t = torch.cat((cls_tokens, img_t), dim=1)  # [batch, num_patches+1, model_d]
        img_t = self.posi_enc(img_t)  # [batch, num_patches+1, model_d]
        return img_t

def download_and_load_weights(model, filename, repo_id="hiki-t/tf_model_mnist"):
    # Download the file from Hugging Face Hub
    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    # Load the weights
    model.load_state_dict(torch.load(local_path, map_location="cpu", weights_only=True))
    return model

class DigitSequenceDecoder(nn.Module):
    def __init__(self, encoder_dim, hidden_dim, seq_len, num_classes=10):
        super().__init__()
        self.seq_len = seq_len
        self.gru = nn.GRU(encoder_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, encoder_output):
        # encoder_output: [batch, encoder_dim]
        # Repeat encoder output for each step in the sequence
        repeated = encoder_output.unsqueeze(1).repeat(1, self.seq_len, 1)  # [batch, seq_len, encoder_dim]
        out, _ = self.gru(repeated)  # [batch, seq_len, hidden_dim]
        logits = self.fc(out)        # [batch, seq_len, num_classes]
        return logits

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

class MNISTTransformerModel:
    def __init__(self, model_path=".tf_model_mnist/trained_model/", seq_len=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model configuration (should match training config)
        self.config = {
            "patch_size": 7,
            "patch_len": 16,
            "model_d": 128,
            "num_layers": 2,
            "dropout": 0.1,
            "num_heads": 2,
        }
        
        # Initialize models
        self.preprocess_model = PatchEmbedder(
            self.config["patch_size"], 
            self.config["model_d"], 
            self.config["patch_len"]
        )
        self.tf_model = TransformerEncoder(
            self.config["model_d"], 
            self.config["num_layers"], 
            self.config["dropout"], 
            self.config["num_heads"]
        )
        self.lin_class_m = nn.Linear(self.config["model_d"], 10)
        
        # Load trained weights if available
        self.model_path = model_path
        self.load_models()
                
        # Move to device
        self.preprocess_model.to(self.device)
        self.tf_model.to(self.device)
        self.lin_class_m.to(self.device)
        
        # Transform for input images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        ### this is for multi-digits
        self.seq_len = seq_len # or any fixed number of digits
        self.decoder = DigitSequenceDecoder(self.config["model_d"], 128, self.seq_len)
        self.decoder.eval()
        self.decoder.to(self.device)

    def set_eval_mode(self):
        """Set all models to evaluation mode (frozen)"""
        self.preprocess_model.eval()
        self.tf_model.eval()
        self.lin_class_m.eval()
        self.decoder.eval()
    
    def set_train_mode(self):
        """Set all models to training mode (unfrozen)"""
        self.preprocess_model.train()
        self.tf_model.train()
        self.lin_class_m.train()
        self.decoder.train()

    def load_models(self):
        """Load trained model weights"""
        # try:
        #     if Path(f"{self.model_path}/pytorch_pp_model.bin").exists():
        #         self.preprocess_model.load_state_dict(
        #             torch.load(f"{self.model_path}/pytorch_pp_model.bin", map_location=self.device, weights_only=True)
        #         )
        #         self.tf_model.load_state_dict(
        #             torch.load(f"{self.model_path}/pytorch_tf_model.bin", map_location=self.device, weights_only=True)
        #         )
        #         self.lin_class_m.load_state_dict(
        #             torch.load(f"{self.model_path}/pytorch_class_lin_model.bin", map_location=self.device, weights_only=True)
        #         )
        #         print("Loaded trained model weights successfully!")
        #     else:
        #         print("No trained weights found. Using untrained models.")
        # except Exception as e:
        #     print(f"Error loading models: {e}")
        #     print("Using untrained models.")
        self.preprocess_model = download_and_load_weights(self.preprocess_model, "pytorch_pp_model.bin")
        self.tf_model = download_and_load_weights(self.tf_model, "pytorch_tf_model.bin")
        self.lin_class_m = download_and_load_weights(self.lin_class_m, "pytorch_class_lin_model.bin")
        print("Loaded trained model weights from hf repo successfully!")
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # print(f"DEBUG: preprocess_image input type: {type(image)}")
        
        if isinstance(image, str):
            # Load image from file path
            image = Image.open(image).convert('L')  # Convert to grayscale
            # print(f"DEBUG: Loaded from file, size: {image.size}, mode: {image.mode}")
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            # print(f"DEBUG: Input array shape: {image.shape}, dtype: {image.dtype}")
            if image.ndim == 3 and image.shape[2] == 4:
                # RGBA image (from canvas) - convert to grayscale
                image = Image.fromarray(image, 'RGBA').convert('L')
                # print(f"DEBUG: Converted RGBA to grayscale, size: {image.size}")
            elif image.ndim == 3 and image.shape[2] == 3:
                # RGB image - convert to grayscale
                image = Image.fromarray(image, 'RGB').convert('L')
                # print(f"DEBUG: Converted RGB to grayscale, size: {image.size}")
            else:
                # Already grayscale or single channel
                image = Image.fromarray(image).convert('L')
                # print(f"DEBUG: Converted to grayscale, size: {image.size}")
        elif isinstance(image, Image.Image):
            # Ensure PIL image is converted to grayscale
            # print(f"DEBUG: Converting PIL image from mode {image.mode} to grayscale")
            image = image.convert('L')
        
        # Resize to 28x28 if needed
        if image.size != (28, 28):
            # print(f"DEBUG: Resizing from {image.size} to (28, 28)")
            image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # ... after image is grayscale and 28x28
        image = ImageOps.invert(image)

        # Apply transformations
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension

        # ### 
        # ### for debugging
        # ### 

        # # ... inside preprocess_image(self, image):
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Get current time and format it
        # save_dir = ".test"
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, f"processed_input_{timestamp}.png")
        # image.save(save_path)
        # save_path_np = os.path.join(save_dir, f"processed_input_{timestamp}.npy")
        # # Convert PIL image to numpy array and save
        # np.save(save_path_np, np.array(image))
        # # save preprocessed and torch image
        # save_path_t = os.path.join(save_dir, f"processed_tensor_{timestamp}.npy")
        # # Move tensor to CPU and convert to numpy, then save
        # np.save(save_path_t, image_tensor.cpu().numpy())

        # print(f"DEBUG: Final tensor shape: {image_tensor.shape}")
        return image_tensor.to(self.device)

    def preprocess_tensor(self, img_tensor):
        """Preprocess tensor for model input"""
        # Ensure tensor is on the right device
        img_tensor = img_tensor.to(self.device)
        
        return img_tensor

    def predict(self, image):
        """Predict digit from image"""
        # print(f"DEBUG: predict() called with image type: {type(image)}")
        
        with torch.no_grad():
            # Preprocess image
            img_tensor = self.preprocess_image(image)
            # print(f"DEBUG: After preprocess_image, tensor shape: {img_tensor.shape}")
            
            # Patch embedding
            img_embedded = self.preprocess_model(img_tensor)
            # print(f"DEBUG: After patch embedding, shape: {img_embedded.shape}")
            
            # Transformer encoding
            transformer_output = self.tf_model(img_embedded)
            # print(f"DEBUG: After transformer, shape: {transformer_output.shape}")
            
            # Classification
            cls_output = transformer_output[:, 0, :]  # Take CLS token
            # print(f"DEBUG: CLS token shape: {cls_output.shape}")

            ### this is for single digit image
            # logits = self.lin_class_m(cls_output)
            # # print(f"DEBUG: Logits shape: {logits.shape}")
            
            # # Get probabilities and prediction
            # probabilities = torch.softmax(logits, dim=1)
            # prediction = torch.argmax(logits, dim=1).item()
            # confidence = probabilities[0, prediction].item()

            # print(f"DEBUG: Prediction: {prediction}, Confidence: {confidence:.3f}")
            
            # return {
            #     "prediction": prediction,
            #     "confidence": confidence,
            #     "probabilities": probabilities[0].cpu().numpy().tolist()
            # }

            ### this is for multiple digit image
            seq_logits = self.decoder(cls_output)     # [batch, seq_len, 10]
            # seq_logits: [batch, seq_len, 10]
            probabilities = torch.softmax(seq_logits, dim=2)  # [batch, seq_len, 10]
            predictions = torch.argmax(seq_logits, dim=2)  # [batch, seq_len]
            # For batch=0 (first sample)
            confidences = probabilities[0, range(seq_logits.shape[1]), predictions[0]]  # [seq_len]
            # This gives a tensor of length seq_len, each entry is the confidence for that digit

            return {
                "prediction": predictions[0].cpu().tolist(),      # e.g., [1, 2, 3]
                "confidence": confidences,                       # e.g., [0.98, 0.87, 0.92]
                "probabilities": probabilities[0].cpu().tolist()  # shape: [seq_len, 10]
            }

    def predict_batch(self, images):
        """Predict digits from multiple images"""
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results 