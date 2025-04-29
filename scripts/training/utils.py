#!/usr/bin/env python3
# Common utilities for training models
# This script contains utility functions used across different training scripts

import os
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F

def seed_everything(seed):
    """Set random seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_versioned_directory(base_dir, prefix="run"):
    """
    Create a versioned directory for storing run artifacts.
    
    Args:
        base_dir: Base directory to create the versioned directory in
        prefix: Prefix for the directory name (e.g., "run", "model", etc.)
        
    Returns:
        tuple: (path, run_number) - path to the created directory and its number
    """
    # Check if there are any existing run directories
    existing_runs = [d for d in os.listdir(base_dir) 
                      if os.path.isdir(os.path.join(base_dir, d)) 
                      and d.startswith(prefix)]
    
    # Extract run numbers and determine the next one
    run_numbers = [int(d.replace(prefix, "")) for d in existing_runs if d[len(prefix):].isdigit()]
    next_run = 1 if not run_numbers else max(run_numbers) + 1
    
    # Create the new run directory
    run_dir = os.path.join(base_dir, f"{prefix}{next_run}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Created versioned directory: {run_dir}")
    return run_dir, next_run

def calculate_iou(preds, targets, threshold=0.5, num_classes=1):
    """
    Calculate Intersection over Union (IoU) for segmentation tasks.
    
    Args:
        preds: Predicted segmentation masks
        targets: Ground truth segmentation masks
        threshold: Threshold for binary segmentation (ignored for multi-class)
        num_classes: Number of classes (1 for binary segmentation)
        
    Returns:
        float: Mean IoU across batch and classes
    """
    if num_classes == 1:
        # Binary segmentation case
        batch_size = preds.size(0)
        iou_sum = 0.0
        
        # Move to CPU for calculation
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()
        
        # Apply threshold for binary case
        preds = (preds > threshold).float()
        
        for i in range(batch_size):
            pred = preds[i].view(-1)
            target = targets[i].view(-1)
            
            # Calculate intersection and union
            intersection = torch.sum(pred * target)
            union = torch.sum(pred) + torch.sum(target) - intersection
            
            # Calculate IoU for this sample
            iou = intersection / union if union > 0 else torch.tensor(1.0)
            iou_sum += iou.item()
        
        return iou_sum / batch_size
    else:
        # Multi-class segmentation case
        batch_size = preds.size(0)
        iou_sum = 0.0
        valid_classes = 0
        
        # Move to CPU for calculation
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()
        
        for c in range(num_classes):
            # Create binary masks for this class
            pred_c = (preds == c).float().view(batch_size, -1)
            target_c = (targets == c).float().view(batch_size, -1)
            
            # Calculate intersection and union for each sample
            intersection = torch.sum(pred_c * target_c, dim=1)
            union = torch.sum(pred_c, dim=1) + torch.sum(target_c, dim=1) - intersection
            
            # Calculate per-sample IoU and average
            valid_samples = union > 0
            if valid_samples.sum() > 0:
                iou_c = (intersection[valid_samples] / union[valid_samples]).mean().item()
                iou_sum += iou_c
                valid_classes += 1
        
        # Return mean IoU across all valid classes
        return iou_sum / valid_classes if valid_classes > 0 else 0.0

def plot_learning_curves(epochs, train_losses, val_losses, val_metrics, save_path=None):
    """
    Plot learning curves for training and validation.
    
    Args:
        epochs: List of epoch numbers
        train_losses: List of training losses
        val_losses: List of validation losses
        val_metrics: List of validation metrics (e.g., IoU)
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot metrics (IoU)
    ax2.plot(epochs, val_metrics, 'g-', label='Validation IoU')
    ax2.set_title('IoU Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Learning curves saved to {save_path}")
    
    plt.close()

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in segmentation tasks.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t is the predicted probability of the target class.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Can be a tensor for class weights
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Get class probabilities
        if inputs.dim() > 2:
            # In case of typical segmentation task with dimensions [B,C,H,W]
            inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, inputs.size(1))
            targets = targets.view(-1)
        
        # Get log softmax outputs to have better numerical stability
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Get probability of the target class
        target_probs = probs.gather(1, targets.unsqueeze(1))
        target_probs = target_probs.view(-1)
        
        # Calculate focal weights
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            focal_weight = focal_weight * alpha_weight
        
        # Calculate final loss
        loss = -focal_weight * log_probs.gather(1, targets.unsqueeze(1)).view(-1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Basic U-Net model architecture
class UNet(nn.Module):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    (Ronneberger et al., 2015)
    
    A basic U-Net implementation that is commonly used for segmentation tasks.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self._double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._double_conv(512, 1024)
        
        # Decoder (upsampling)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._double_conv(1024, 512)  # 1024 = 512 + 512 (skip connection)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._double_conv(512, 256)   # 512 = 256 + 256 (skip connection)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._double_conv(256, 128)   # 256 = 128 + 128 (skip connection)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._double_conv(128, 64)    # 128 = 64 + 64 (skip connection)
        
        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def _double_conv(self, in_channels, out_channels):
        """Double convolutional block with batch normalization and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder with skip connections
        d4 = self.up4(b)
        # Handle potential size mismatch with skip connections
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # Final layer
        return self.final(d1) 