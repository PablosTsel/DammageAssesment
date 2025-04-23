#!/usr/bin/env python3
# Building Detection Model - Stage 1 of the two-stage approach
# This script trains a U-Net architecture that only detects building footprints
# without classifying damage levels

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
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
import shutil

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

# Import the UNet architecture and helper functions
from scripts.training.localization_model import (
    UNet, seed_everything, create_versioned_directory, 
    calculate_iou, plot_learning_curves
)

# Constants
BUILDING_LABEL = 1
NON_BUILDING_LABEL = 0

# Dataset class for binary building segmentation
class XBDBinaryBuildingDataset(Dataset):
    """
    Dataset for binary building segmentation that:
    - Loads pre-disaster satellite images
    - Converts building polygons from JSON files to binary masks (building vs. non-building)
    - Returns (image, mask) pairs for training a segmentation model
    """
    def __init__(self, 
                 root_dir,
                 image_size=256,
                 use_xy=True,
                 max_samples=None,
                 augment=False):
        """
        Initialize the binary building segmentation dataset.
        
        Args:
            root_dir: Directory where xBD data is stored
            image_size: Size of the input/output images (square)
            use_xy: Use 'xy' coordinates if True; else use 'lng_lat'
            max_samples: Optional limit on the number of samples
            augment: If True, applies data augmentation
        """
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size
        self.coord_key = "xy" if use_xy else "lng_lat"
        self.max_samples = max_samples
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
            
        print(f"Loaded {len(self.samples)} samples for binary building segmentation")
    
    def _gather_samples(self):
        """
        Parse the dataset directory to find all valid pre-disaster images with 
        building footprint annotations.
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
                
                pre_json_path = os.path.join(labels_dir, pre_label_file)
                pre_img_path = os.path.join(images_dir, pre_img_name)
                
                # Skip if files don't exist
                if not (os.path.isfile(pre_json_path) and os.path.isfile(pre_img_path)):
                    continue
                
                samples.append({
                    "img_path": pre_img_path,
                    "pre_json_path": pre_json_path
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
                        
                        pre_json_path = os.path.join(labels_dir, pre_label_file)
                        pre_img_path = os.path.join(images_dir, pre_img_name)
                        
                        # Skip if files don't exist
                        if not (os.path.isfile(pre_json_path) and os.path.isfile(pre_img_path)):
                            continue
                        
                        samples.append({
                            "img_path": pre_img_path,
                            "pre_json_path": pre_json_path,
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
        Returns the pre-disaster image and corresponding binary building mask.
        """
        item = self.samples[idx]
        img_path = item["img_path"]
        pre_json_path = item["pre_json_path"]
        
        try:
            # Load the pre-disaster image
            img = Image.open(img_path).convert("RGB")
            original_size = img.size  # (width, height)
            
            # Create an empty mask (0 = background/non-building)
            mask = Image.new("L", original_size, NON_BUILDING_LABEL)
            draw = ImageDraw.Draw(mask)
            
            # Load pre-disaster JSON to get building polygon geometries
            with open(pre_json_path, 'r') as f:
                pre_json_data = json.load(f)
            
            # Extract building polygons from the pre-disaster JSON
            pre_feats = pre_json_data.get("features", {}).get(self.coord_key, [])
            for feat in pre_feats:
                wkt_str = feat.get("wkt", None)
                
                if not wkt_str:
                    continue
                
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
                
                # Draw the polygon filled with the building label
                draw.polygon(coords, fill=BUILDING_LABEL)
            
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
            mask_tensor = torch.from_numpy(np.array(mask)).float().unsqueeze(0)  # Add channel dimension
            
            return img_tensor, mask_tensor
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return placeholder tensors in case of error
            img_tensor = torch.zeros(3, self.image_size, self.image_size)
            mask_tensor = torch.zeros(1, self.image_size, self.image_size)
            return img_tensor, mask_tensor


def visualize_binary_predictions(model, dataset, device, num_samples=4, save_path=None):
    """
    Visualize binary building segmentation predictions.
    """
    model.eval()
    # Select random indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # Configure the plot
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    # Track total IoU for binary segmentation
    total_iou = 0
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get sample
            image, mask = dataset[idx]
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            
            # Get prediction
            pred = model(image)
            
            # Apply sigmoid to get probability map
            pred = torch.sigmoid(pred)
            
            # Create binary prediction (threshold at 0.5)
            pred_binary = (pred > 0.5).float()
            
            # Calculate IoU for this sample
            pred_flat = pred_binary.view(-1).cpu().numpy()
            mask_flat = mask.view(-1).cpu().numpy()
            iou = jaccard_score(mask_flat, pred_flat, average='binary')
            total_iou += iou
            
            # Convert to numpy for visualization
            pred_binary = pred_binary.squeeze().cpu().numpy()
            mask = mask.squeeze().cpu().numpy()
            
            # Denormalize image for display
            image = image.squeeze().cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = np.transpose(image, (1, 2, 0))
            image = image * std + mean
            image = np.clip(image, 0, 1)
            
            # Plot in the current row
            axes[i, 0].imshow(image)
            axes[i, 0].set_title("Pre-disaster Image")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title("Ground Truth Buildings")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_binary, cmap='gray')
            axes[i, 2].set_title(f"Predicted Buildings (IoU: {iou:.4f})")
            axes[i, 2].axis('off')
    
    # Calculate average IoU
    avg_iou = total_iou / num_samples
    plt.figtext(0.5, 0.01, f"Average IoU: {avg_iou:.4f}", ha="center", 
                fontsize=14, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Binary prediction visualization saved to {save_path}")
    
    plt.close()
    
    return avg_iou


def main():
    """
    Main function to run the binary building segmentation training pipeline.
    """
    # Set multiprocessing start method to 'spawn' to avoid CUDA initialization errors
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'")
    except RuntimeError:
        print("Multiprocessing start method already set to 'spawn' or could not be set")
        pass
        
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_everything(42)
    
    # Hyperparameters & settings
    root_dir = os.path.join(project_root, "data", "xBD")
    batch_size = 16
    lr = 0.0002
    num_epochs = 25
    val_ratio = 0.2
    image_size = 256
    
    # Create output directory for binary segmentation model
    output_dir = os.path.join(project_root, "output", "binary_building")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create versioned run directory
    run_dir, run_num = create_versioned_directory(output_dir, prefix="binary_run")
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
    print("Initializing XBDBinaryBuildingDataset for training...")
    full_dataset = XBDBinaryBuildingDataset(
        root_dir=root_dir,
        image_size=image_size,
        use_xy=True,
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
    
    # Initialize model - binary segmentation has 1 output channel
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    # Binary segmentation uses BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    
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
            outputs = model(images)
            
            # Compute loss
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
                outputs = model(images)
                
                # Compute loss
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Apply sigmoid and threshold to get binary predictions
                preds = (torch.sigmoid(outputs) > 0.5).float()
                
                # Calculate IoU for binary segmentation
                batch_iou = calculate_iou(preds, masks, threshold=0.5)
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
            visualize_binary_predictions(model, full_dataset, device, num_samples=4, save_path=vis_save_path)
    
    # Copy the best model as binary_best.pt
    if best_model_path and os.path.exists(best_model_path):
        binary_best_path = os.path.join(output_dir, "binary_building_best.pt")
        shutil.copy2(best_model_path, binary_best_path)
        print(f"Best model copied to {binary_best_path}")
    
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