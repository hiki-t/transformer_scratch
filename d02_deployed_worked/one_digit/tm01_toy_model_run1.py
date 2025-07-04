import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from einops import rearrange, reduce, repeat
from huggingface_hub import HfApi
from pathlib import Path

import wandb
import math

#####################################################################################################################
#####################################################################################################################

###
### load pretrained model and dataset
###

def load_data(batch_size):

    # Step 1: Unfold the image into patches
    transform_test = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ])

    # data augumentations
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)), # random rotation and translation (shift)
        # transforms.RandomRotation(10), # additional random rotation
        transforms.RandomCrop(28, padding=4), # random crop with padding (simulates small shifts)
        transforms.ToTensor(), # standard preprocessing
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_trainset = datasets.MNIST(root='./test_data', train=True, download=True, transform=transform)
    mnist_testset = datasets.MNIST(root="./train_data", train=False, download=True, transform=transform_test)

    train_dataloader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(mnist_testset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

#####################################################################################################################
#####################################################################################################################

###
### all functions and models
###

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

class PatchEmbedding(nn.Module):
    def __init__(self, patch_pixels, d_model):
        super().__init__()
        self.proj = nn.Linear(patch_pixels, d_model)
    def forward(self, x):
        # x: [batch, num_patches, patch_pixels, channels]
        x = x.squeeze(-1)  # remove channel dim if 1
        return self.proj(x)

class EncoderLayer(nn.Module):
    def __init__(self, model_d, dropout, num_heads=1, mlp_ratio=4):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(model_d, num_heads)
        self.norm1 = nn.LayerNorm(model_d)
        self.norm2 = nn.LayerNorm(model_d)
        self.dropout = nn.Dropout(dropout)
        # self.lin = nn.Linear(model_d, model_d)
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
        # out = self.lin(x)  # [batch, num_classes]
        out = self.mlp(x)
        # x = x + self.dropout(out)
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

#####################################################################################################################
#####################################################################################################################

# train and validaiton

### train tf_model
def train_model( data_loader, preprocess_model, tf_model, lin_class_m, configs):

    lr = configs["lr"]
    num_epochs = configs["epochs"]
    epoch_out_freq = configs["epoch_out_freq"]

    # Set up loss function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(tf_model.parameters(), lr=lr)

    tf_model.train()
    for epoch in range(num_epochs):

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for (img_t, label) in data_loader:
            # img_t: [batch_size, 1, 28, 28]

            # do step1-3
            img_t = preprocess_model(img_t)

            # step4: single head model process
            out = tf_model(img_t) # [batch=64, patch=16, emb=128]

            # step5: convert out to cls_out
            # Assume model(img_t) returns [batch, num_patches+1, emb]
            cls_out = out[:, 0, :]  # [batch, emb]
            logits = lin_class_m(cls_out)  # [batch, num_classes]
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * img_t.size(0)

            # Compute accuracy
            preds = logits.argmax(dim=1)  # [batch]
            total_correct += (preds == label).sum().item()
            total_samples += img_t.size(0)

        # Compute average loss and accuracy
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        # Log to wandb
        wandb.log({"train/loss": avg_loss, "train/accuracy": avg_acc, "epoch": epoch+1})

        # output
        if (epoch+1) % epoch_out_freq == 0:
            print(f"Train: Epoch{epoch+1} Average loss: {avg_loss:.4f}")

### validation on trained model
def get_validation( data_loader, preprocess_model, tf_model, lin_class_m ):

    # Set up loss function
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    tf_model.eval()
    with torch.no_grad():
        for (img_t, label) in data_loader:
            # img_t: [batch_size, 1, 28, 28]

            # do step1-3
            img_t = preprocess_model(img_t)

            # step4: single head model process
            out = tf_model(img_t) # [batch=64, patch=16, emb=128]

            # step5: convert out to cls_out
            # Assume model(img_t) returns [batch, num_patches+1, emb]
            cls_out = out[:, 0, :]  # [batch, emb]
            logits = lin_class_m(cls_out)  # [batch, num_classes]
            loss = criterion(logits, label)
            
            total_loss += loss.item() * img_t.size(0)

            # Compute accuracy
            preds = logits.argmax(dim=1)
            total_correct += (preds == label).sum().item()
            total_samples += img_t.size(0)

    # Compute average loss and accuracy
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    # Log to wandb
    wandb.log({"val/loss": avg_loss, "val/accuracy": accuracy})

    # output
    print(f"Test: Average loss: {avg_loss:.4f}")
    print(f"Test: Accuracy: {accuracy*100:.2f}%")

#####################################################################################################################
#####################################################################################################################

def main(configs):

    wandb.init(
        project="vit_mnist_from_scratch",  # Change to your project name
        entity="htsujimu-ucl",  # <--- set this to your username or team name
        config=configs
    )

    # 1. Load data
    print("load datasets"); 
    train_dataloader, test_dataloader = load_data(configs["batch_size"])

    # 2. Build models
    print("build models"); 
    preprocess_model = PatchEmbedder(configs["patch_size"], configs["model_d"], configs["patch_len"])
    tf_model = TransformerEncoder(configs["model_d"], configs["num_layers"], configs["dropout"], configs["num_heads"])
    lin_class_m = nn.Linear(configs["model_d"], 10)

    # 3. Training parameters
    params = {
        "lr": configs["lr"],
        "epochs": configs["epochs"],
        "epoch_out_freq": configs["epoch_out_freq"],
    }

    # 4. Load trained weights if already exists
    if Path("./tf_model_mnist/trained_model/pytorch_pp_model.bin").exists():
        print("loading trained weights")
        preprocess_model.load_state_dict(torch.load("./tf_model_mnist/trained_model/pytorch_pp_model.bin", weights_only=True))
        tf_model.load_state_dict(torch.load("./tf_model_mnist/trained_model/pytorch_tf_model.bin", weights_only=True))
        lin_class_m.load_state_dict(torch.load("./tf_model_mnist/trained_model/pytorch_class_lin_model.bin", weights_only=True))
    else:
        print("No trained model, start from scratch.")

    # 5. Train and validate
    print("training models"); 
    train_model(train_dataloader, preprocess_model, tf_model, lin_class_m, params)

    print("get validation results"); 
    get_validation(test_dataloader, preprocess_model, tf_model, lin_class_m)

    # 6. Save trained weights and push to HF
    print("save trained models")
    torch.save(preprocess_model.state_dict(), "./tf_model_mnist/trained_model/pytorch_pp_model.bin")
    torch.save(tf_model.state_dict(), "./tf_model_mnist/trained_model/pytorch_tf_model.bin")
    torch.save(lin_class_m.state_dict(), "./tf_model_mnist/trained_model/pytorch_class_lin_model.bin")

    print("push trained models to hf")
    # Set your repo name and user/org
    repo_id = configs["hf_repo"]
    api = HfApi()
    
    # If the repo doesn't exist, create it (only needs to be done once)
    api.create_repo(repo_id=repo_id, exist_ok=True)

    # push trained models to hf
    api.upload_file(
        path_or_fileobj="./tf_model_mnist/trained_model/pytorch_pp_model.bin",
        path_in_repo="pytorch_pp_model.bin",
        repo_id=repo_id,
    )
    api.upload_file(
        path_or_fileobj="./tf_model_mnist/trained_model/pytorch_tf_model.bin",
        path_in_repo="pytorch_tf_model.bin",
        repo_id=repo_id,
    )
    api.upload_file(
        path_or_fileobj="./tf_model_mnist/trained_model/pytorch_class_lin_model.bin",
        path_in_repo="pytorch_class_lin_model.bin",
        repo_id=repo_id,
    )

    wandb.finish()

if __name__ == "__main__":
    configs = {
        "batch_size": 64,
        "patch_size": 7,
        "patch_len": 16,
        "model_d": 128,
        "num_layers": 2,
        "dropout": 0.1,
        "num_heads": 2,
        "lr": 3e-4,
        "epochs": 12,
        "epoch_out_freq": 1, 
        "hf_repo": "hiki-t/tf_model_mnist", # use your own repo
    }
    main(configs)
