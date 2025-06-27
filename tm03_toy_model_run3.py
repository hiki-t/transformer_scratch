"""
CTC Sliding Window Model for Multi-Digit Recognition
Based on Connectionist Temporal Classification approach
Reference: https://distill.pub/2017/ctc/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import json
import wandb
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_utils import MNISTTransformerModel

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class SlidingWindowSequenceModel(nn.Module):
    """Simple sequence model for multi-digit recognition"""
    
    def __init__(self, backbone_model, seq_length=4, num_classes=10):
        super().__init__()
        self.backbone = backbone_model
        self.seq_length = seq_length
        self.num_classes = num_classes
        
        # Freeze backbone (as requested) - access individual components
        for param in self.backbone.preprocess_model.parameters():
            param.requires_grad = False
        for param in self.backbone.tf_model.parameters():
            param.requires_grad = False
        for param in self.backbone.lin_class_m.parameters():
            param.requires_grad = False
        if hasattr(self.backbone, 'decoder'):
            for param in self.backbone.decoder.parameters():
                param.requires_grad = False
            
        # Simple sequence decoder
        self.sequence_decoder = nn.Linear(128, seq_length * num_classes)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, 1, height, width] - Single digit images
        Returns:
            logits: [batch_size, seq_length, num_classes] - Logits for each position
        """
        batch_size = x.size(0)
        
        # Process through backbone
        img_embedded = self.backbone.preprocess_model(x)
        transformer_output = self.backbone.tf_model(img_embedded)
        
        # Take CLS token for each digit
        cls_output = transformer_output[:, 0, :]  # [batch_size, 128]
        
        # Sequence decoder
        sequence_logits = self.sequence_decoder(cls_output)  # [batch_size, seq_length * num_classes]
        
        # Reshape to [batch_size, seq_length, num_classes]
        logits = sequence_logits.view(batch_size, self.seq_length, self.num_classes)
        
        return logits

def create_sequence_batches(img_batch, labels, seq_length=4):
    """
    Create sequence batches from single digits
    """
    batch_size = img_batch.size(0)
    num_sequences = batch_size // seq_length
    
    if num_sequences == 0:
        return None, None
    
    # Reshape to create sequences
    sequences = img_batch[:num_sequences * seq_length].view(num_sequences, seq_length, 1, 28, 28)
    sequence_labels = labels[:num_sequences * seq_length].view(num_sequences, seq_length)
    
    return sequences, sequence_labels

def evaluate_sliding_window(model, dataloader, device, seq_length=4):
    """Evaluate sequence model"""
    model.eval()
    total_correct = 0
    total_sequences = 0
    
    with torch.no_grad():
        for img, label in dataloader:
            sequences, sequence_labels = create_sequence_batches(
                img, label, seq_length=seq_length
            )
            
            if sequences is None:
                continue
                
            sequences = sequences.to(device)
            sequence_labels = sequence_labels.to(device)
            
            # Forward pass
            logits = model(sequences.view(-1, 1, 28, 28))
            logits = logits.view(-1, seq_length, 10)
            predictions = torch.argmax(logits, dim=2)
            
            # Compare with ground truth
            correct = (predictions == sequence_labels).all(dim=1)
            total_correct += correct.sum().item()
            total_sequences += len(sequence_labels)
    
    accuracy = total_correct / total_sequences if total_sequences > 0 else 0
    return accuracy

class SequenceTrainingConfig:
    """Configuration for sequence training"""
    def __init__(self):
        # Model parameters
        self.seq_length = 4  # Start with 4, can increase to 8, 16, 32
        self.batch_size = 64
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        self.num_epochs = 20
        
        # Training settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wandb = True
        self.save_dir = "./trained_models"
        
        # Data settings
        self.data_dir = ".data"
        self.train_split = 0.8
        self.val_split = 0.2

class SequenceTrainer:
    """Trainer for sequence recognition"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss function (cross entropy for sequence classification)
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer (only for sequence decoder)
        self.optimizer = optim.Adam(
            self.model.sequence_decoder.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (img, label) in enumerate(dataloader):
            # Create sequence batches
            sequences, sequence_labels = create_sequence_batches(
                img, label, seq_length=self.config.seq_length
            )
            
            if sequences is None:
                continue
                
            # Move to device
            sequences = sequences.to(self.device)
            sequence_labels = sequence_labels.to(self.device)
            
            # Forward pass
            logits = self.model(sequences.view(-1, 1, 28, 28))  # [num_sequences * seq_len, seq_len, num_classes]
            logits = logits.view(-1, self.config.seq_length, self.config.num_classes)
            
            # Calculate loss
            # Reshape for cross entropy: [batch * seq_len, num_classes]
            logits_flat = logits.view(-1, self.config.num_classes)
            targets_flat = sequence_labels.view(-1)
            
            loss = self.criterion(logits_flat, targets_flat)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        return epoch_loss / num_batches if num_batches > 0 else 0
    
    def train(self, train_dataloader, val_dataloader):
        """Complete training loop"""
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Sequence length: {self.config.seq_length}")
        print(f"Batch size: {self.config.batch_size}")
        
        best_val_acc = 0
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss = self.train_epoch(train_dataloader, epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            val_accuracy = evaluate_sliding_window(
                self.model, val_dataloader, self.device, self.config.seq_length
            )
            self.val_accuracies.append(val_accuracy)
            
            # Learning rate scheduling
            self.scheduler.step(train_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_accuracy': val_accuracy,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                self.save_model(f"best_model_seq{self.config.seq_length}.pth")
                print(f"New best model saved! Val Accuracy: {val_accuracy:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies
        }
    
    def save_model(self, filename):
        """Save model checkpoint"""
        save_path = Path(self.config.save_dir) / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies
        }, save_path)
        print(f"Model saved to {save_path}")

def setup_training_environment(config):
    """Setup training environment"""
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Setup wandb if enabled
    if config.use_wandb:
        wandb.init(
            project="mnist-sequence-recognition",
            config=vars(config),
            name=f"seq_len_{config.seq_length}_training"
        )
    
    # Save config (convert device to string for JSON serialization)
    config_dict = vars(config).copy()
    config_dict['device'] = str(config.device)
    
    config_path = Path(config.save_dir) / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Training environment setup complete")
    print(f"Device: {config.device}")
    print(f"Save directory: {config.save_dir}")
    
    return config

def load_data(batch_size, data_dir=".data"):
    """Load MNIST dataset"""
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    # Split training data into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_dataloader, val_dataloader, test_dataloader

def visualize_sequence_predictions(model, dataloader, device, seq_length=4, num_examples=5):
    """Visualize sequence predictions"""
    model.eval()
    
    with torch.no_grad():
        for img, label in dataloader:
            sequences, sequence_labels = create_sequence_batches(
                img, label, seq_length=seq_length
            )
            
            if sequences is None:
                continue
                
            for i in range(min(num_examples, len(sequences))):
                seq_imgs = sequences[i].to(device)  # [seq_length, 1, 28, 28]
                seq_labels = sequence_labels[i]     # [seq_length]
                
                # Forward pass
                logits = model(seq_imgs)  # [seq_length, seq_length, num_classes]
                predictions = torch.argmax(logits, dim=2)  # [seq_length, seq_length]
                
                # Visualize
                fig, axes = plt.subplots(1, seq_length, figsize=(seq_length * 2, 2))
                if seq_length == 1:
                    axes = [axes]
                
                for j in range(seq_length):
                    digit_img = seq_imgs[j].squeeze().cpu()
                    axes[j].imshow(digit_img, cmap='gray')
                    axes[j].set_title(f"True: {seq_labels[j].item()}")
                    axes[j].axis('off')
                
                # Get predictions for this sequence
                pred_sequence = predictions[0].cpu().numpy()  # Take first prediction
                plt.suptitle(f"Predicted: {pred_sequence}")
                plt.tight_layout()
                plt.show()
            
            break

def main():
    """Main training script for sliding window sequence"""
    
    # Setup configuration
    config = SequenceTrainingConfig()
    config = setup_training_environment(config)
    
    # Load data
    print("Loading data...")
    train_dataloader, val_dataloader, test_dataloader = load_data(config.batch_size)
    
    # Initialize backbone model (frozen)
    print("Initializing model...")
    backbone = MNISTTransformerModel()
    
    # Create sequence model
    model = SlidingWindowSequenceModel(backbone, seq_length=config.seq_length)
    
    # Initialize trainer
    trainer = SequenceTrainer(model, config)
    
    # Start training
    print("Starting sequence training...")
    history = trainer.train(train_dataloader, val_dataloader)
    
    # Final evaluation
    print("Final evaluation...")
    final_val_loss, final_val_acc = trainer.train_losses[-1], trainer.val_accuracies[-1]
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    
    # Test evaluation
    test_accuracy = evaluate_sliding_window(model, test_dataloader, config.device, config.seq_length)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Visualize some predictions
    print("Visualizing predictions...")
    visualize_sequence_predictions(model, test_dataloader, config.device, config.seq_length)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracies'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(Path(config.save_dir) / 'training_history.png')
    plt.show()
    
    print("Training complete!")
    
    if config.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 