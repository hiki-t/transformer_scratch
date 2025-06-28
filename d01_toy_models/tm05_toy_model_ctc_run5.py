
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import random
from pathlib import Path
from huggingface_hub import HfApi
import wandb
import tqdm

from model_utils import MNISTTransformerModel

###
### setup device
###

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###
### set parameters
###

batch_size = 64
seq_length=4
slide_size=8
num_epochs=3
lr=3e-4

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

    train_dataloader = DataLoader(mnist_trainset, batch_size=batch_size, num_workers=6, pin_memory=True, shuffle=True)
    test_dataloader = DataLoader(mnist_testset, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader

### 
### custom functions
### 

def encode_patch(one_image, model):  # Input: 28x28 image patch
    img_embedded = model.preprocess_model(one_image)
    transformer_output = model.tf_model(img_embedded)
    cls_output = transformer_output[:, 0, :]    
    logits = model.lin_class_m(cls_output) # [1, 10] or digit class size
    # logits = enc_model(one_image.unsqueeze(0).unsqueeze(0))  # [1, 1, 28, 28]
    probs = torch.softmax(logits, dim=1) 
    pred = probs.argmax(dim=1)
    confidence = probs[0, pred].item()
    return logits, pred, confidence, cls_output

def slide_seq_img_encode(seq_img, model, slide_size):
    # seq img size expects [patch_size(height), patch_size(width)] no batch, no channel
    w_size = seq_img.shape[-1]
    patch_size = seq_img.shape[0]
    window_len = int((w_size-patch_size)/slide_size)

    pred_per_digit_logit = []
    pred_per_digit_pred = []
    pred_per_digit_conf = []
    pred_per_digit_cls_out = []

    for j in range(window_len):
        x_position = j*slide_size
        crop_img = seq_img[:, x_position:x_position+patch_size].unsqueeze(0).unsqueeze(0)

        logits, pred, confidence, cls_output = encode_patch(crop_img, model)

        pred_per_digit_logit.append(logits)
        pred_per_digit_pred.append(pred)
        pred_per_digit_conf.append(confidence)
        pred_per_digit_cls_out.append(cls_output)

    pred_per_digit_logit = torch.stack(pred_per_digit_logit).squeeze(1)  # shape: [T, num_classes]
    pred_per_digit_cls_out = torch.stack(pred_per_digit_cls_out).squeeze(1)  # shape: [T, num_classes]

    return pred_per_digit_logit, pred_per_digit_pred, pred_per_digit_conf, pred_per_digit_cls_out

class SequenceDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # BiRNN output

    def forward(self, logits_seq):  # logits_seq: [B, T, C]
        out, _ = self.rnn(logits_seq)  # [B, T, 2*H]
        out = self.fc(out)             # [B, T, num_classes]
        return out

def add_blank_class_logits(logits_10, threshold=0.5, blank_boost=5.0):
    """
    Takes [T, 10] logits and returns [T, 11] logits with a manually added blank class.

    Args:
        logits_10: Tensor of shape [T, 10]
        threshold: If max prob < threshold, boost blank class
        blank_boost: How much to boost blank class logit when needed

    Returns:
        logits_11: Tensor of shape [T, 11]
    """
    probs = F.softmax(logits_10, dim=1)
    max_conf, _ = probs.max(dim=1)  # [T]

    # Default blank logit = small value
    blank_logits = torch.full((logits_10.size(0), 1), fill_value=-10.0, device=logits_10.device)

    # Where confidence is low, boost blank logit
    high_blank_indices = max_conf < threshold
    blank_logits[high_blank_indices] = blank_boost

    # Concatenate to original logits → [T, 11]
    logits_11 = torch.cat([logits_10, blank_logits], dim=1)
    return logits_11


def create_sequence_image(img_batch, labels, seq_length=4, canvas_height=28):
    """
    Create a single image containing multiple digits with random positioning
    
    Args:
        img_batch: [batch_size, 1, 28, 28] - batch of single digit images
        labels: [batch_size] - corresponding labels
        seq_length: number of digits to combine
        canvas_height: height of the output image
    
    Returns:
        sequence_images: [num_sequences, 1, canvas_height, canvas_width] - combined images
        sequence_labels: [num_sequences, seq_length] - corresponding labels
    """
    batch_size = img_batch.size(0)
    num_sequences = batch_size // seq_length
    
    # Calculate canvas width (enough space for seq_length digits with gaps)
    max_gap = 1  # Maximum gap between digits
    min_gap = -4  # Minimum gap (negative for overlap)
    canvas_width = seq_length * 28 + (seq_length - 1) * max_gap + 25  # Extra padding
    
    sequence_images = []
    sequence_labels = []
    
    for seq_idx in range(num_sequences):
        # Extract sequence
        start_idx = seq_idx * seq_length
        end_idx = start_idx + seq_length
        
        seq_images = img_batch[start_idx:end_idx].squeeze(1)  # [seq_length, 28, 28]
        seq_labels = labels[start_idx:end_idx]  # [seq_length]
        
        # Create canvas
        canvas = torch.ones((canvas_height, canvas_width), device=device)*img_batch.min()
        
        # Position first digit
        pos = random.randint(0, 10)  # Random starting position
        
        # Place digits with random gaps
        for i in range(seq_length):
            # Add random gap (negative for overlap, positive for gap)
            if i > 0:
                gap = random.randint(min_gap, max_gap)
                pos = max(0, pos + gap)
            
            # Place digit
            canvas[:, pos:pos+28] += seq_images[i]
            pos += 28
        
        # # Normalize and clamp
        canvas = torch.clamp(canvas, img_batch.min(), img_batch.max())
        
        # Add channel dimension and batch dimension
        canvas = canvas.unsqueeze(0).unsqueeze(0)  # [1, 1, canvas_height, canvas_width]
        
        sequence_images.append(canvas)
        sequence_labels.append(seq_labels)
    
    # Stack all sequences
    sequence_images = torch.cat(sequence_images, dim=0)  # [num_sequences, 1, canvas_height, canvas_width]
    sequence_labels = torch.stack(sequence_labels)  # [num_sequences, seq_length]
    
    return sequence_images, sequence_labels

def visualize_sequence_image(sequence_image, sequence_labels, seq_idx=0):
    """Visualize a sequence image with its labels"""
    img = sequence_image[seq_idx].squeeze(0)  # Remove channel dimension
    labels = sequence_labels[seq_idx]
    
    plt.figure(figsize=(12, 4))
    plt.imshow(img, cmap='gray')
    plt.title(f'Sequence {seq_idx}: Labels = {labels.tolist()}')
    plt.colorbar()
    plt.axis('off')
    plt.show()

def ctc_greedy_decode(log_probs, blank=10):
    """ Greedy CTC decoder that removes blanks and repeated labels. """
    pred = torch.argmax(log_probs, dim=2).squeeze(1).tolist()  # [T]
    decoded = []
    prev = None
    for p in pred:
        if p != blank and p != prev:
            decoded.append(p)
        prev = p
    return decoded  # list of integers

### 
### load models and data
### 

# load trained encoder model
encoder = MNISTTransformerModel()
# load decoder model
decoder = SequenceDecoder(input_size=128, hidden_size=128, num_classes=11).to(device) # 10 digits

# Load trained weights if already exists
if Path("./.tf_model_mnist/trained_model3/pytorch_dec_model_part3.bin").exists():
    print("loading trained weights")
    decoder.load_state_dict(torch.load("./.tf_model_mnist/trained_model3/pytorch_dec_model_part3.bin", weights_only=True))
else:
    print("No trained model, start from scratch.")

encoder.preprocess_model.to(device)
encoder.tf_model.to(device)
encoder.lin_class_m.to(device)

# load data
train_dataloader, test_dataloader = load_data(batch_size)

### 
### set train and validation
### 

# Set up loss function
ctc_loss = nn.CTCLoss(blank=10)
optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr)

encoder.preprocess_model.eval()
encoder.tf_model.eval()
encoder.lin_class_m.eval()
decoder.train()

wandb.init(
    project="vit_mnist_from_scratch3", 
    entity="htsujimu-ucl",  # <--- set this to your username or team name
)

step = 0
for epoch in range(num_epochs):

    train_dataloader = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=False)

    for (img, label) in train_dataloader:
        
        total_digits = 0
        correct_digits = 0
        total_sequences = 0
        correct_sequences = 0

        img = img.to(device)
        label = label.to(device)

        # Create sequence images
        sequence_images, sequence_labels = create_sequence_image(img, label, seq_length=seq_length)

        sequence_images = sequence_images.to(device)
        sequence_labels = sequence_labels.to(device)

        batch_losses = []
        # Visualize a few sequences
        for i in range(sequence_images.shape[0]): # this will loop within a batch=16

            # visualize_sequence_image(sequence_images, sequence_labels, i)
            seq_img = sequence_images[i][0]
            seq_label = sequence_labels[i]

            seq_img = seq_img.to(device)
            seq_label = seq_label.to(device)

            _, _, pred_conf_10, pred_cls_out_10  = slide_seq_img_encode(seq_img, encoder, slide_size)
            conf_tensor = torch.tensor(pred_conf_10).unsqueeze(1)  # shape: [T, 1]
            conf_tensor = conf_tensor.to(device)

            weighted_cls = conf_tensor * pred_cls_out_10
            decoder_output = decoder(weighted_cls.unsqueeze(0))  # [1, T, 10]
            log_probs = F.log_softmax(decoder_output, dim=2).transpose(0, 1)  # [T, 1, C]
            log_probs = log_probs.to(device)

            loss = ctc_loss(
                log_probs,
                seq_label, 
                input_lengths=torch.tensor([weighted_cls.size(0)], dtype=torch.long, device=device),
                target_lengths=torch.tensor([seq_label.size(0)], dtype=torch.long, device=device)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### this is for prediction
            # After getting log_probs:
            decoded = ctc_greedy_decode(log_probs, blank=10)
            true = seq_label.tolist()

            # Per-digit accuracy
            min_len = min(len(decoded), len(true))
            correct_digits += sum([p == t for p, t in zip(decoded[:min_len], true[:min_len])])
            total_digits += len(true)

            # Per-sequence accuracy
            if decoded == true:
                correct_sequences += 1
            total_sequences += 1

            batch_losses.append(loss)
        
        total_loss = torch.stack(batch_losses).mean()
        
        digit_acc = correct_digits / total_digits * 100
        sequence_acc = correct_sequences / total_sequences * 100
        
        # Log to wandb
        wandb.log({"train/loss": total_loss, "train/accuracy-digit": digit_acc, "train/accuracy-seq": sequence_acc}, step = step)
        step += 1

    print(f"Epoch {epoch} - Loss: {total_loss.item():.4f} | Digit Acc: {digit_acc:.2f}% | Sequence Acc: {sequence_acc:.2f}%")

# 6. Save trained weights and push to HF
print("save trained models")
torch.save(decoder.state_dict(), "./.tf_model_mnist/trained_model3/pytorch_dec_model_part3.bin")

print("push trained models to hf")
# Set your repo name and user/org
repo_id = "hiki-t/tf_model_mnist"
api = HfApi()

# If the repo doesn't exist, create it (only needs to be done once)
api.create_repo(repo_id=repo_id, exist_ok=True)

# push trained models to hf
api.upload_file(
    path_or_fileobj="./.tf_model_mnist/trained_model3/pytorch_dec_model_part3.bin",
    path_in_repo="pytorch_pp_model_part3.bin",
    repo_id=repo_id,
)

### test

total_digits = 0
correct_digits = 0
total_sequences = 0
correct_sequences = 0

encoder.preprocess_model.eval()
encoder.tf_model.eval()
encoder.lin_class_m.eval()
decoder.eval()

with torch.no_grad():
    for (img, label) in test_dataloader:        
        # Create sequence images
        sequence_images, sequence_labels = create_sequence_image(img, label, seq_length=seq_length)
        
        batch_losses = []
        # Visualize a few sequences
        for i in range(sequence_images.shape[0]): # this will loop within a batch=16

            # visualize_sequence_image(sequence_images, sequence_labels, i)
            seq_img = sequence_images[i][0]
            seq_label = sequence_labels[i]

            _, _, pred_conf_10, pred_cls_out_10  = slide_seq_img_encode(seq_img, encoder, slide_size)
            conf_tensor = torch.tensor(pred_conf_10).unsqueeze(1)  # shape: [T, 1]
            weighted_cls = conf_tensor * pred_cls_out_10
            decoder_output = decoder(weighted_cls.unsqueeze(0))  # [1, T, 10]
            log_probs = F.log_softmax(decoder_output, dim=2).transpose(0, 1)  # [T, 1, C]

            loss = ctc_loss(
                log_probs,
                seq_label, 
                input_lengths=torch.tensor([weighted_cls.size(0)], dtype=torch.long),
                target_lengths=torch.tensor([seq_label.size(0)], dtype=torch.long)
            )

            ### this is for prediction
            # After getting log_probs:
            decoded = ctc_greedy_decode(log_probs, blank=10)
            true = seq_label.tolist()

            # Per-digit accuracy
            min_len = min(len(decoded), len(true))
            correct_digits += sum([p == t for p, t in zip(decoded[:min_len], true[:min_len])])
            total_digits += len(true)

            # Per-sequence accuracy
            if decoded == true:
                correct_sequences += 1
            total_sequences += 1

            batch_losses.append(loss)
        
        total_loss = torch.stack(batch_losses).mean()

digit_acc = correct_digits / total_digits * 100
sequence_acc = correct_sequences / total_sequences * 100
print(f"Test - Loss: {total_loss.item():.4f} | Digit Acc: {digit_acc:.2f}% | Sequence Acc: {sequence_acc:.2f}%")

# Log to wandb
wandb.log({"val/loss": total_loss, "val/accuracy-digit": digit_acc, "val/accuracy-seq": sequence_acc})

wandb.finish()
