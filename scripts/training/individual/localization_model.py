#!/usr/bin/env python3
# xBD Building Localization Model
# This script implements a U-Net architecture for building footprint segmentation
# from pre-disaster satellite imagery

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset, WeightedRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import jaccard_score
import random
from datetime import datetime
import time
import torch.nn.functional as F
import json
from PIL import Image, ImageDraw
import torchvision.transforms as T
from shapely import wkt
from shapely.geometry import Polygon
import cv2
import shutil

# Constants
BUILDING_LABEL = 1
NON_BUILDING_LABEL = 0

# Add Focal Loss implementation for better handling of class imbalance
class FocalLoss(nn.Module):
    """
    Focal Loss implementation for multi-class segmentation with class imbalance.
    Based on https://arxiv.org/pdf/1708.02002.pdf
    
    This loss function down-weights well-classified examples and focuses training on hard examples.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Calculate focal loss.
        Args:
            inputs: Predictions from model (N, C, H, W) as logits
            targets: Ground truth labels (N, H, W) as class indices
        """
        # Apply log softmax to get log probabilities
        log_softmax = F.log_softmax(inputs, dim=1)
        
        # Gather the log softmax using the target indices
        # This gets the log probability of the correct class for each pixel
        batch_size = inputs.size(0)
        loss = torch.zeros_like(targets, dtype=inputs.dtype).to(inputs.device)
        
        for cls in range(inputs.size(1)):
            # Create a mask for this class
            target_mask = (targets == cls)
            if target_mask.sum() > 0:
                # Extract log probabilities for this class
                class_log_prob = log_softmax[:, cls, :, :]
                
                # Compute probability of correct class
                class_prob = torch.exp(class_log_prob)
                
                # Focal loss weighting: (1-pt)^gamma
                modulating_factor = (1.0 - class_prob) ** self.gamma
                
                # Apply focal loss to the masked positions
                class_loss = -modulating_factor * class_log_prob
                
                # Apply alpha weighting if specified
                if self.alpha is not None:
                    alpha_weight = self.alpha[cls]
                    class_loss = alpha_weight * class_loss
                
                # Update the loss for this class
                loss[target_mask] = class_loss[target_mask]
                
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

# Dataset class for building segmentation
class XBDBuildingSegDataset(Dataset):
    """
    Dataset for building segmentation that:
    - Loads pre-disaster satellite images
    - Converts building polygons from JSON files to multi-class masks with damage levels
    - Returns (image, mask) pairs for training a segmentation model
    """
    def __init__(self, 
                 root_dir,
                 image_size=256,
                 use_xy=True,
                 max_samples=None,
                 flat_structure=False,
                 augment=False):
        """
        Initialize the building segmentation dataset.
        
        Args:
            root_dir: Directory where xBD data is stored
            image_size: Size of the input/output images (square)
            use_xy: Use 'xy' coordinates if True; else use 'lng_lat'
            max_samples: Optional limit on the number of samples
            flat_structure: Whether the folder structure is flat
            augment: If True, applies data augmentation
        """
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size
        self.coord_key = "xy" if use_xy else "lng_lat"
        self.max_samples = max_samples
        self.flat_structure = flat_structure
        self.augment = augment
        
        # Define damage class mapping (0 is background/non-building)
        self.damage_class_map = {
            'no-damage': 1,
            'minor-damage': 2,
            'major-damage': 3,
            'destroyed': 4,
            'un-classified': 1  # Map unclassified to no-damage
        }
        
        # Transforms for the input images
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        
        # Gather all samples from the dataset directory
        self.samples = self._gather_samples()
        
        # Limit the number of samples if specified
        if self.max_samples is not None and len(self.samples) > self.max_samples:
            self.samples = random.sample(self.samples, self.max_samples)
            
        print(f"Loaded {len(self.samples)} samples for building segmentation")
    
    def _gather_samples(self):
        """
        Parse the dataset directory to find all valid pre-disaster images with 
        both pre and post-disaster JSON labels for damage classification.
        """
        samples = []
        
        # Check for flat structure first
        images_dir = os.path.join(self.root_dir, "images")
        labels_dir = os.path.join(self.root_dir, "labels")
        
        if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
            # Process flat directory structure
            print(f"Detected flat structure at {self.root_dir}")
            
            # Find all pre-disaster JSON files
            pre_label_files = [f for f in os.listdir(labels_dir) if f.endswith("_pre_disaster.json")]
            
            for pre_label_file in pre_label_files:
                # Extract base ID from filename
                base_id = pre_label_file.replace("_pre_disaster.json", "")
                pre_img_name = base_id + "_pre_disaster.png"
                post_label_file = base_id + "_post_disaster.json"
                
                pre_json_path = os.path.join(labels_dir, pre_label_file)
                pre_img_path = os.path.join(images_dir, pre_img_name)
                post_json_path = os.path.join(labels_dir, post_label_file)
                
                # Skip if files don't exist
                if not (os.path.isfile(pre_json_path) and os.path.isfile(pre_img_path) and 
                        os.path.isfile(post_json_path)):
                    continue
                
                samples.append({
                    "img_path": pre_img_path,
                    "pre_json_path": pre_json_path,
                    "post_json_path": post_json_path
                })
                
        else:
            # Process hierarchical directory structure
            try:
                # List all disaster directories
                disasters = [d for d in os.listdir(self.root_dir)
                            if os.path.isdir(os.path.join(self.root_dir, d))
                            and d.lower() != "spacenet_gt"]
                
                print(f"Found {len(disasters)} disaster folders")
                
                # Process each disaster folder
                for disaster in disasters:
                    disaster_dir = os.path.join(self.root_dir, disaster)
                    images_dir = os.path.join(disaster_dir, "images")
                    labels_dir = os.path.join(disaster_dir, "labels")
                    
                    # Skip if directories don't exist
                    if not (os.path.isdir(images_dir) and os.path.isdir(labels_dir)):
                        print(f"Warning: Missing images or labels directory for disaster: {disaster}")
                        continue
                    
                    # Find all pre-disaster JSON files
                    pre_label_files = [f for f in os.listdir(labels_dir) if f.endswith("_pre_disaster.json")]
                    print(f"Disaster {disaster}: Found {len(pre_label_files)} label files")
                    
                    # Process each label file
                    for pre_label_file in pre_label_files:
                        base_id = pre_label_file.replace("_pre_disaster.json", "")
                        pre_img_name = base_id + "_pre_disaster.png"
                        post_label_file = base_id + "_post_disaster.json"
                        
                        pre_json_path = os.path.join(labels_dir, pre_label_file)
                        pre_img_path = os.path.join(images_dir, pre_img_name)
                        post_json_path = os.path.join(labels_dir, post_label_file)
                        
                        # Skip if files don't exist
                        if not (os.path.isfile(pre_json_path) and os.path.isfile(pre_img_path) and 
                                os.path.isfile(post_json_path)):
                            continue
                        
                        samples.append({
                            "img_path": pre_img_path,
                            "pre_json_path": pre_json_path,
                            "post_json_path": post_json_path,
                            "disaster": disaster
                        })
            except Exception as e:
                print(f"Error gathering samples: {e}")
        
        return samples
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        Returns the pre-disaster image and corresponding multi-class damage mask.
        """
        item = self.samples[idx]
        img_path = item["img_path"]
        pre_json_path = item["pre_json_path"]
        post_json_path = item["post_json_path"]
        
        try:
            # Load the pre-disaster image
            img = Image.open(img_path).convert("RGB")
            original_size = img.size  # (width, height)
            
            # Create an empty mask (0 = background/non-building)
            mask = Image.new("L", original_size, 0)
            draw = ImageDraw.Draw(mask)
            
            # First, create a mapping of building UIDs to damage classes from post-disaster JSON
            with open(post_json_path, 'r') as f:
                post_json_data = json.load(f)
            
            # Map building IDs to damage classes
            building_damage = {}
            post_feats = post_json_data.get("features", {}).get(self.coord_key, [])
            for feat in post_feats:
                uid = feat.get("properties", {}).get("uid", None)
                damage_type = feat.get("properties", {}).get("subtype", "").lower()
                if uid and damage_type in self.damage_class_map:
                    building_damage[uid] = self.damage_class_map[damage_type]
            
            # Now load pre-disaster JSON to get building polygon geometries
            with open(pre_json_path, 'r') as f:
                pre_json_data = json.load(f)
            
            # Extract building polygons from the pre-disaster JSON
            pre_feats = pre_json_data.get("features", {}).get(self.coord_key, [])
            for feat in pre_feats:
                uid = feat.get("properties", {}).get("uid", None)
                wkt_str = feat.get("wkt", None)
                
                if not wkt_str:
                    continue
                
                # Determine damage class for this building
                # Default to no-damage (class 1) if not found in post-disaster data
                damage_class = building_damage.get(uid, 1)
                
                # Parse the WKT string to get the polygon
                polygon = wkt.loads(wkt_str)
                
                # Convert polygon to a list of (x, y) tuples for PIL's polygon drawing
                if hasattr(polygon, 'exterior'):
                    # For simple polygons with exterior coordinates
                    coords = list(polygon.exterior.coords)
                else:
                    # For multipolygons or other geometries, try to extract coordinates
                    try:
                        coords = list(polygon.coords)
                    except:
                        # Skip polygons that can't be processed
                        continue
                
                # Draw the polygon filled with the appropriate damage class value
                draw.polygon(coords, fill=damage_class)
            
            # Apply augmentation if enabled
            if self.augment:
                # Same random seed for both transforms to apply consistent augmentation
                seed = np.random.randint(2147483647)
                random.seed(seed)
                torch.manual_seed(seed)
                
                # Simple rotation/flip augmentation
                if random.random() > 0.5:
                    img = T.functional.hflip(img)
                    mask = T.functional.hflip(mask)
                if random.random() > 0.5:
                    img = T.functional.vflip(img)
                    mask = T.functional.vflip(mask)
                if random.random() > 0.5:
                    angle = random.choice([90, 180, 270])
                    img = T.functional.rotate(img, angle)
                    mask = T.functional.rotate(mask, angle)
            
            # Apply transforms
            img_tensor = self.image_transform(img)
            
            # Resize the mask and convert to tensor
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            mask_tensor = torch.from_numpy(np.array(mask)).long()  # Use long tensor for class indices
            
            return img_tensor, mask_tensor
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return placeholder tensors in case of error
            img_tensor = torch.zeros(3, self.image_size, self.image_size)
            mask_tensor = torch.zeros(self.image_size, self.image_size, dtype=torch.long)
            return img_tensor, mask_tensor


# U-Net model definition
class DoubleConv(nn.Module):
    """Double convolutional block as used in U-Net architecture."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation of buildings.
    """
    def __init__(self, in_channels=3, out_channels=5):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder (upsampling)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # 1024 = 512 + 512 (skip connection)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)   # 512 = 256 + 256 (skip connection)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)   # 256 = 128 + 128 (skip connection)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)    # 128 = 64 + 64 (skip connection)
        
        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
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
        up4 = self.up4(b)
        # Handle potential size mismatch with skip connections
        if up4.shape != e4.shape[2:]:
            up4 = F.interpolate(up4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([up4, e4], dim=1))
        
        up3 = self.up3(d4)
        if up3.shape != e3.shape[2:]:
            up3 = F.interpolate(up3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([up3, e3], dim=1))
        
        up2 = self.up2(d3)
        if up2.shape != e2.shape[2:]:
            up2 = F.interpolate(up2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))
        
        up1 = self.up1(d2)
        if up1.shape != e1.shape[2:]:
            up1 = F.interpolate(up1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))
        
        # Final layer - return logits directly (no softmax activation)
        # This change makes it compatible with both CrossEntropyLoss and our FocalLoss
        return self.final(d1)


# Functions for training and evaluation
def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


def create_versioned_directory(base_path, prefix="ALONE_loc_run"):
    """
    Create a versioned directory with incremented run number if base exists.
    This helps keep multiple training runs organized.
    
    Args:
        base_path: Base directory path
        prefix: Prefix for the directory name
    
    Returns:
        full_path: Path to the created directory
        i: Version number
    """
    i = 1
    while True:
        dir_name = f"{prefix}{i}"
        full_path = os.path.join(base_path, dir_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)
            return full_path, i
        i += 1


def calculate_iou(pred, target, threshold=0.5, num_classes=5):
    """
    Calculate mean Intersection over Union (IoU) score across all classes.
    For multi-class segmentation, computes IoU for each class and returns the mean.
    
    Args:
        pred: Predicted segmentation map (logits or class indices)
        target: Ground truth segmentation map
        threshold: Threshold for binary segmentation
        num_classes: Number of classes for multi-class segmentation
    
    Returns:
        Mean IoU across all classes
    """
    # For binary segmentation (building vs non-building)
    # This part is kept for backward compatibility
    if pred.shape[1] == 1 and target.shape[1] == 1:
        # Apply threshold to obtain binary masks
        pred_binary = (pred > threshold).float()
        
        # Flatten the tensors for simple calculation
        pred_flat = pred_binary.view(-1).cpu().numpy()
        target_flat = target.view(-1).cpu().numpy()
        
        # Calculate IoU using scikit-learn's implementation
        iou = jaccard_score(target_flat, pred_flat, average='binary')
        return iou
    
    # For multi-class segmentation
    # If pred is logits, convert to class indices
    if pred.shape[1] > 1 and pred.dim() == 4:  # [batch_size, num_classes, height, width]
        # Get the predicted class using argmax
        _, pred_classes = torch.max(pred, dim=1)  # Shape: [batch_size, height, width]
    else:
        pred_classes = pred  # Assume pred already contains class indices
    
    # Similarly for target
    if target.shape[1] > 1 and target.dim() == 4:
        _, target_classes = torch.max(target, dim=1)
    else:
        target_classes = target
    
    # Ensure both are of the right shape
    if pred_classes.dim() == 3:  # [batch_size, height, width]
        pass  # Already in correct format
    elif pred_classes.dim() == 2:  # [height, width]
        pred_classes = pred_classes.unsqueeze(0)  # Add batch dimension
    
    if target_classes.dim() == 3:
        pass  # Already in correct format
    elif target_classes.dim() == 2:
        target_classes = target_classes.unsqueeze(0)
    
    # Calculate IoU for each class
    batch_size = pred_classes.size(0)
    total_iou = 0.0
    
    # Process each sample in the batch separately
    for b in range(batch_size):
        sample_iou = 0.0
        num_valid_classes = 0
        
        # Calculate IoU for each class
        for cls in range(num_classes):
            # Extract binary masks for the current class
            pred_mask = (pred_classes[b] == cls).cpu().numpy().flatten()
            target_mask = (target_classes[b] == cls).cpu().numpy().flatten()
            
            # Skip if this class is not present in the ground truth
            if np.sum(target_mask) == 0:
                continue
            
            # Calculate intersection and union
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            
            # Calculate IoU for this class
            if union == 0:
                iou = 0.0
            else:
                iou = intersection / union
            
            sample_iou += iou
            num_valid_classes += 1
        
        # Calculate mean IoU for this sample
        if num_valid_classes > 0:
            sample_iou /= num_valid_classes
        
        total_iou += sample_iou
    
    # Return mean IoU across all samples in the batch
    return total_iou / batch_size


def plot_learning_curves(epochs, train_losses, val_losses, val_ious, save_path):
    """Plot and save training metrics visualizations."""
    plt.figure(figsize=(12, 5))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker='o', color='blue')
    plt.plot(epochs, val_losses, label="Val Loss", marker='o', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot validation IoU
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_ious, label="Val IoU", marker='o', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("IoU Score")
    plt.title("Validation IoU")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Learning curves plot saved to {save_path}")


def visualize_predictions(model, dataset, device, num_samples=4, save_path=None):
    """
    Visualize model predictions and calculate per-class IoU metrics.
    Saves the original image, ground truth mask, and predicted mask side by side.
    Also calculates and displays IoU for each damage class to evaluate classification accuracy.
    """
    model.eval()
    # Select random indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # Define colors for each damage class
    # Format: [R, G, B]
    damage_colors = [
        [0, 0, 0],       # Background (black)
        [0, 255, 0],     # No damage (green)
        [255, 255, 0],   # Minor damage (yellow)
        [255, 165, 0],   # Major damage (orange)
        [255, 0, 0]      # Destroyed (red)
    ]
    
    # Class names for display
    class_names = ['Background', 'No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
    
    # Configure the plot
    fig = plt.figure(figsize=(15, 6 * num_samples))
    
    # Track per-class IoU stats
    class_pixels_total = {cls: 0 for cls in range(5)}
    class_pixels_correct = {cls: 0 for cls in range(5)}
    class_pixels_pred = {cls: 0 for cls in range(5)}
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get sample
            image, mask = dataset[idx]
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            
            # Get prediction
            pred = model(image)
            
            # Convert logits to class predictions
            pred = F.softmax(pred, dim=1).squeeze().cpu().numpy()  # Shape: [num_classes, H, W]
            
            # Convert to class indices using argmax along the class dimension
            pred_class = np.argmax(pred, axis=0)  # Shape: [H, W]
            
            # Get ground truth mask
            mask = mask.squeeze().cpu().numpy()
            
            # Create colored visualizations
            colored_pred = np.zeros((pred_class.shape[0], pred_class.shape[1], 3), dtype=np.uint8)
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            
            # Color each class in the prediction and ground truth
            for class_idx in range(len(damage_colors)):
                colored_pred[pred_class == class_idx] = damage_colors[class_idx]
                colored_mask[mask == class_idx] = damage_colors[class_idx]
            
            # Calculate per-class metrics for this sample
            for cls in range(5):
                # Ground truth pixels for this class
                gt_pixels = (mask == cls)
                # Predicted pixels for this class
                pred_pixels = (pred_class == cls)
                
                # True positives (correctly classified)
                true_pos = np.logical_and(gt_pixels, pred_pixels).sum()
                
                # Update totals
                class_pixels_total[cls] += gt_pixels.sum()
                class_pixels_correct[cls] += true_pos
                class_pixels_pred[cls] += pred_pixels.sum()
            
            # Denormalize image for display
            image = image.squeeze().cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = np.transpose(image, (1, 2, 0))
            image = image * std + mean
            image = np.clip(image, 0, 1)
            
            # Plot images on the top row
            plt.subplot(num_samples, 3, i*3 + 1)
            plt.imshow(image)
            plt.title("Pre-disaster Image")
            plt.axis('off')
            
            plt.subplot(num_samples, 3, i*3 + 2)
            plt.imshow(colored_mask)
            plt.title("Ground Truth Mask")
            plt.axis('off')
            
            plt.subplot(num_samples, 3, i*3 + 3)
            plt.imshow(colored_pred)
            plt.title("Predicted Damage Classes")
            plt.axis('off')
    
    # Calculate IoU for each class
    class_iou = {}
    for cls in range(5):
        if class_pixels_total[cls] == 0:
            class_iou[cls] = float('nan')  # No ground truth pixels for this class
        else:
            union = class_pixels_total[cls] + class_pixels_pred[cls] - class_pixels_correct[cls]
            if union > 0:
                class_iou[cls] = class_pixels_correct[cls] / union
            else:
                class_iou[cls] = float('nan')
    
    # Add a text box with per-class IoU results
    plt.figtext(0.5, 0.01, f"Per-Class IoU Metrics:", ha="center", fontsize=14, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    text = ""
    for cls in range(5):
        iou_value = class_iou[cls]
        iou_str = f"{iou_value:.4f}" if not np.isnan(iou_value) else "N/A"
        text += f"{class_names[cls]}: {iou_str}   "
    
    plt.figtext(0.5, 0.005, text, ha="center", fontsize=12)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave room for the text at the bottom
    
    if save_path:
        plt.savefig(save_path)
        print(f"Prediction visualization saved to {save_path}")
    
    # Print per-class IoU to console as well
    print("\nPer-Class IoU Metrics:")
    for cls in range(5):
        iou_value = class_iou[cls]
        iou_str = f"{iou_value:.4f}" if not np.isnan(iou_value) else "N/A"
        print(f"  {class_names[cls]}: {iou_str}")
    
    plt.close()
    
    return class_iou


def main():
    """
    Main function to run the building localization training pipeline.
    Handles data loading, model training, evaluation, and saving results.
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_everything(42)
    
    # Get project root directory
    project_root = os.path.abspath(os.path.dirname(__file__))
    # Go up two levels from scripts/training to the project root
    project_root = os.path.abspath(os.path.join(project_root, "..", ".."))
    
    # Hyperparameters & settings
    root_dir = os.path.join(project_root, "data", "xBD")
    batch_size = 16
    lr = 0.0002
    num_epochs = 12  # Changed to 12 epochs as requested
    val_ratio = 0.2
    image_size = 256
    num_classes = 5  # Class 0: background, 1: no-damage, 2: minor-damage, 3: major-damage, 4: destroyed
    
    # Create output directory
    output_dir = os.path.join(project_root, "output", "ALONE_localization")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create versioned run directory
    run_dir, run_num = create_versioned_directory(output_dir)
    model_dir = os.path.join(run_dir, "models")
    viz_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Save configuration
    config = {
        "timestamp": timestamp,
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_epochs": num_epochs,
        "val_ratio": val_ratio,
        "image_size": image_size,
        "num_classes": num_classes
    }
    
    with open(os.path.join(run_dir, f"config_run{run_num}.txt"), "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize dataset
    print("Initializing XBDBuildingSegDataset for training...")
    full_dataset = XBDBuildingSegDataset(
        root_dir=root_dir,
        image_size=image_size,
        use_xy=True,
        max_samples=None,
        augment=True
    )
    
    # Create train-val split
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split = int(np.floor(val_ratio * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Calculate class frequencies from training dataset
    print("Calculating class frequencies for weighting...")
    class_counts = torch.zeros(num_classes)
    
    # Keep track of damage class presence in each sample for weighted sampling
    sample_weights = torch.ones(len(train_indices))
    
    for i, idx in enumerate(tqdm(train_indices, desc="Counting class pixels")):
        _, mask = full_dataset[idx]
        
        # Count pixels of each class
        for c in range(num_classes):
            class_count = (mask == c).sum().item()
            class_counts[c] += class_count
            
            # Give higher weights to samples containing damaged buildings
            if c >= 2 and class_count > 0:  # If sample has minor, major, or destroyed damage
                # Scale the weight based on the damage level and amount
                damage_weight = (c / num_classes) * 10  # Higher damage classes get higher weights
                sample_weights[i] = max(sample_weights[i], damage_weight)
    
    # Calculate class weights inversely proportional to frequencies
    total_pixels = class_counts.sum()
    # Normalized class frequencies
    class_frequencies = class_counts / total_pixels
    print(f"Class distribution: {class_frequencies.tolist()}")
    
    # Calculate inverse frequency weights and normalize
    # Adding epsilon (1e-6) to prevent division by zero
    inverse_freq_weights = 1.0 / (class_frequencies + 1e-6)
    # Normalize weights to make their mean equal to 1
    inverse_freq_weights = inverse_freq_weights / inverse_freq_weights.mean()
    
    # Cap weights to prevent extremely high values
    inverse_freq_weights = torch.clamp(inverse_freq_weights, min=0.1, max=10.0)
    
    # Move to device
    class_weights = inverse_freq_weights.to(device)
    
    print(f"Calculated class weights: {class_weights.tolist()}")
    
    # Create a weighted sampler for training data to balance class representation
    # This will oversample images containing rare damage classes
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_indices),
        replacement=True
    )
    
    print("Using weighted sampler to balance training batches")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use our weighted sampler instead of shuffle
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = UNet(in_channels=3, out_channels=num_classes).to(device)
    
    # Use Focal Loss instead of Cross Entropy Loss for better handling of class imbalance
    # The gamma parameter controls the down-weighting of well-classified examples
    gamma = 2.0  # Recommended value from the paper
    criterion = FocalLoss(gamma=gamma, alpha=class_weights)
    
    # Log the loss function configuration
    print(f"Using Focal Loss with gamma={gamma} and class weights={class_weights.tolist()}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Training metrics tracking
    epochs_list = []
    train_losses = []
    val_losses = []
    val_ious = []
    best_iou = 0.0
    best_model_path = None
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        
        # Training phase
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]", leave=True)
        for images, masks in train_pbar:
            images = images.to(device)
            # Masks are now class indices (0-4), no need to convert
            masks = masks.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass - outputs shape: [batch_size, num_classes, H, W]
            outputs = model(images)  # Now outputs logits directly
            
            # Compute loss using Focal Loss
            loss = criterion(outputs, masks)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate average training loss
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [VAL]", leave=True)
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = model(images)  # Now outputs logits directly
                
                # Compute loss
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Get predicted class for each pixel
                _, preds = torch.max(outputs, dim=1)
                
                # Calculate IoU for multi-class segmentation
                batch_iou = calculate_iou(preds, masks, num_classes=num_classes)
                val_iou += batch_iou
                
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{batch_iou:.4f}"})
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        val_losses.append(avg_val_loss)
        val_ious.append(avg_val_iou)
        epochs_list.append(epoch + 1)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Calculate elapsed time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val IoU: {avg_val_iou:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save the model if it's the best so far
        if avg_val_iou > best_iou:
            # Delete previous best model file if it exists
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
                print(f"Removed previous best model: {best_model_path}")
            
            best_iou = avg_val_iou
            best_model_path = os.path.join(model_dir, f"best_model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with IoU: {best_iou:.4f}")
            
            # Visualize predictions with best model
            vis_save_path = os.path.join(viz_dir, f"predictions_epoch_{epoch+1}.png")
            visualize_predictions(model, full_dataset, device, num_samples=4, save_path=vis_save_path)
    
    # Copy the best model as unet_best.pt
    if best_model_path and os.path.exists(best_model_path):
        unet_best_path = os.path.join(output_dir, "unet_best.pt")
        shutil.copy2(best_model_path, unet_best_path)
        print(f"Best model copied to {unet_best_path}")
    
    # Create learning curves plot
    curves_save_path = os.path.join(viz_dir, "learning_curves.png")
    plot_learning_curves(
        epochs_list,
        train_losses,
        val_losses,
        val_ious,
        curves_save_path
    )
    
    # Save final metrics
    metrics = {
        "epochs": epochs_list,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_ious": val_ious,
    }
    
    metrics_path = os.path.join(run_dir, f"training_metrics_run{run_num}.txt")
    with open(metrics_path, "w") as f:
        for key, values in metrics.items():
            f.write(f"{key}: {values}\n")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"All artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
