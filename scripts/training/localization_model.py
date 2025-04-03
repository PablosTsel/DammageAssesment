#!/usr/bin/env python3
# xBD Building Localization Model
# This script implements a U-Net architecture for building footprint segmentation
# from pre-disaster satellite imagery

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset
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

# Dataset class for building segmentation
class XBDBuildingSegDataset(Dataset):
    """
    Dataset for building segmentation that:
    - Loads pre-disaster satellite images
    - Converts building polygons from JSON files to binary masks
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
        Parse the dataset directory to find all valid pre-disaster images with JSON labels.
        Returns a list of dictionaries with paths to images and corresponding label files.
        """
        samples = []
        
        # Check for flat structure first
        images_dir = os.path.join(self.root_dir, "images")
        labels_dir = os.path.join(self.root_dir, "labels")
        
        if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
            # Process flat directory structure
            print(f"Detected flat structure at {self.root_dir}")
            
            # Find all pre-disaster JSON files (which contain building polygons)
            label_files = [f for f in os.listdir(labels_dir) if f.endswith("_pre_disaster.json")]
            
            for label_file in label_files:
                # Extract base ID from filename
                base_id = label_file.replace("_pre_disaster.json", "")
                pre_img_name = base_id + "_pre_disaster.png"
                
                pre_json_path = os.path.join(labels_dir, label_file)
                pre_img_path = os.path.join(images_dir, pre_img_name)
                
                # Skip if files don't exist
                if not (os.path.isfile(pre_json_path) and os.path.isfile(pre_img_path)):
                    continue
                
                samples.append({
                    "img_path": pre_img_path,
                    "json_path": pre_json_path
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
                    label_files = [f for f in os.listdir(labels_dir) if f.endswith("_pre_disaster.json")]
                    print(f"Disaster {disaster}: Found {len(label_files)} label files")
                    
                    # Process each label file
                    for label_file in label_files:
                        base_id = label_file.replace("_pre_disaster.json", "")
                        pre_img_name = base_id + "_pre_disaster.png"
                        
                        pre_json_path = os.path.join(labels_dir, label_file)
                        pre_img_path = os.path.join(images_dir, pre_img_name)
                        
                        # Skip if files don't exist
                        if not (os.path.isfile(pre_json_path) and os.path.isfile(pre_img_path)):
                            continue
                        
                        samples.append({
                            "img_path": pre_img_path,
                            "json_path": pre_json_path,
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
        Returns the pre-disaster image and corresponding building mask.
        """
        item = self.samples[idx]
        img_path = item["img_path"]
        json_path = item["json_path"]
        
        try:
            # Load the pre-disaster image
            img = Image.open(img_path).convert("RGB")
            original_size = img.size  # (width, height)
            
            # Create an empty mask of the same size as the original image
            mask = Image.new("L", original_size, 0)
            draw = ImageDraw.Draw(mask)
            
            # Load the JSON data
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # Extract building polygons from the JSON
            feats = json_data.get("features", {}).get(self.coord_key, [])
            for feat in feats:
                wkt_str = feat.get("wkt", None)
                if wkt_str is None:
                    continue
                
                # Parse the WKT string to get the polygon and fill it in the mask
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
                
                # Draw the polygon as filled on the mask (255 for building pixels)
                draw.polygon(coords, fill=255)
            
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
            mask_tensor = torch.from_numpy(np.array(mask)).float() / 255.0
            mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
            
            return img_tensor, mask_tensor
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return placeholder tensors in case of error
            img_tensor = torch.zeros(3, self.image_size, self.image_size)
            mask_tensor = torch.zeros(1, self.image_size, self.image_size)
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
    Based on the original U-Net paper with some modern improvements.
    """
    def __init__(self, in_channels=3, out_channels=1):
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
        
        # Final layer with sigmoid activation for binary segmentation
        return torch.sigmoid(self.final(d1))


# Dice loss for segmentation
class DiceLoss(nn.Module):
    """
    Dice loss for image segmentation.
    Computes the Sørensen-Dice loss between predicted and target masks.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice_score


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


def create_versioned_directory(base_path, prefix="localization_run"):
    """Create a versioned directory to store outputs."""
    i = 1
    while True:
        dir_name = f"{prefix}{i}"
        full_path = os.path.join(base_path, dir_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)
            return full_path, i
        i += 1


def calculate_iou(pred, target, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) score between predicted and target masks.
    Predictions are thresholded to create binary masks.
    """
    # Apply threshold to obtain binary masks
    pred_binary = (pred > threshold).float()
    
    # Flatten the tensors for simple calculation
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    # Calculate IoU using scikit-learn's implementation
    iou = jaccard_score(target_flat, pred_flat, average='binary')
    return iou


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
    Visualize model predictions on a few sample images.
    Saves the original image, ground truth mask, and predicted mask side by side.
    """
    model.eval()
    # Select random indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    plt.figure(figsize=(15, 4 * num_samples))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get sample
            image, mask = dataset[idx]
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            
            # Get prediction
            pred = model(image)
            pred = pred.squeeze().cpu().numpy()
            
            # Convert to binary mask
            pred_binary = (pred > 0.5).astype(np.float32)
            
            # Convert tensors to numpy for visualization
            image = image.squeeze().cpu().numpy()
            mask = mask.squeeze().cpu().numpy()
            
            # Denormalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = np.transpose(image, (1, 2, 0))
            image = image * std + mean
            image = np.clip(image, 0, 1)
            
            # Plot
            plt.subplot(num_samples, 3, i*3 + 1)
            plt.imshow(image)
            plt.title("Pre-disaster Image")
            plt.axis('off')
            
            plt.subplot(num_samples, 3, i*3 + 2)
            plt.imshow(mask, cmap='gray')
            plt.title("Ground Truth Mask")
            plt.axis('off')
            
            plt.subplot(num_samples, 3, i*3 + 3)
            plt.imshow(pred_binary, cmap='gray')
            plt.title("Predicted Mask")
            plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Prediction visualization saved to {save_path}")
    
    plt.close()


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
    num_epochs = 25
    val_ratio = 0.2
    image_size = 256
    
    # Create output directory
    output_dir = os.path.join(project_root, "output", "localization")
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
        "image_size": image_size
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
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    # Loss function
    criterion = DiceLoss()
    
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
            masks = masks.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(images)
            
            # Compute loss
            loss = criterion(preds, masks)
            
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
                preds = model(images)
                
                # Compute loss
                loss = criterion(preds, masks)
                val_loss += loss.item()
                
                # Calculate IoU
                batch_iou = calculate_iou(preds, masks)
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
