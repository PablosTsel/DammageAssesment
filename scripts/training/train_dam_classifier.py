#!/usr/bin/env python3
# Improved Damage Classification Model using Pre and Post Disaster Images
# This script implements a damage classification model that analyzes both pre and post-disaster
# images for significantly improved performance

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

# Import from localization_model 
from scripts.training.individual.localization_model import (
    UNet, seed_everything, create_versioned_directory, 
    calculate_iou, plot_learning_curves, FocalLoss
)

# Damage class mapping
DAMAGE_CLASS_MAP = {
    'no-damage': 1,
    'minor-damage': 2,
    'major-damage': 3,
    'destroyed': 4,
    'un-classified': 1  # Map unclassified to no-damage
}

class AttentionFusion(nn.Module):
    """Attention fusion module to combine pre and post disaster features"""
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, pre_feat, post_feat):
        # Concatenate features along channel dimension
        concat_feat = torch.cat([pre_feat, post_feat], dim=1)
        # Generate attention weights
        attn_weights = self.attention(concat_feat)
        # Apply attention to post features
        weighted_post = post_feat * attn_weights
        # Combine pre and weighted post features
        fused_feat = pre_feat + weighted_post
        return fused_feat

class ImprovedDamageClassifier(nn.Module):
    """
    Improved damage classifier that uses both pre and post disaster images
    with an attention fusion mechanism
    """
    def __init__(self, in_channels=3, out_channels=5):
        super(ImprovedDamageClassifier, self).__init__()
        
        # Pre-disaster branch - full UNet with output channels
        self.pre_encoder = UNet(in_channels=in_channels, out_channels=out_channels)
        
        # Post-disaster branch - full UNet with output channels 
        self.post_encoder = UNet(in_channels=in_channels, out_channels=out_channels)
        
        # Store the output channels for later use
        self.out_channels = out_channels
        
        # Attention fusion module for combining pre and post features
        self.attention_fusion = AttentionFusion(in_channels=out_channels)
        
        # Final convolution layers to produce class predictions
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )
        
    def forward(self, pre_img, post_img):
        # Get outputs directly from both UNets
        pre_features = self.pre_encoder(pre_img)
        post_features = self.post_encoder(post_img)
        
        # Fuse features with attention mechanism
        fused_features = self.attention_fusion(pre_features, post_features)
        
        # Final prediction
        output = self.final_conv(fused_features)
        
        return output

class ImprovedDamageDataset(Dataset):
    """
    Dataset for improved damage classification that:
    - Loads pre-disaster and post-disaster satellite images
    - Creates multi-class masks for damage levels
    - Returns (pre_image, post_image, damage_mask) triplets 
    """
    def __init__(self, 
                 root_dir,
                 image_size=256,
                 use_xy=True,
                 max_samples=None,
                 augment=False):
        """
        Initialize the improved damage classification dataset.
        
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
            
        print(f"Loaded {len(self.samples)} samples for improved damage classification")
    
    def _gather_samples(self):
        """
        Parse the dataset directory to find all valid pre/post-disaster image pairs
        with corresponding label files
        """
        samples = []
        
        # Check for flat structure first
        images_dir = os.path.join(self.root_dir, "images")
        labels_dir = os.path.join(self.root_dir, "labels")
        
        if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
            # Process flat directory structure
            print(f"Detected flat structure at {self.root_dir}")
            
            # Find all post-disaster JSON files (which contain damage labels)
            post_label_files = [f for f in os.listdir(labels_dir) if f.endswith("_post_disaster.json")]
            
            for post_label_file in post_label_files:
                # Extract base ID from filename
                base_id = post_label_file.replace("_post_disaster.json", "")
                post_img_name = base_id + "_post_disaster.png"
                pre_img_name = base_id + "_pre_disaster.png"
                pre_json_path = os.path.join(labels_dir, base_id + "_pre_disaster.json")
                
                post_json_path = os.path.join(labels_dir, post_label_file)
                post_img_path = os.path.join(images_dir, post_img_name)
                pre_img_path = os.path.join(images_dir, pre_img_name)
                
                # Skip if files don't exist
                if not (os.path.isfile(post_json_path) and os.path.isfile(post_img_path) and 
                        os.path.isfile(pre_json_path) and os.path.isfile(pre_img_path)):
                    continue
                
                samples.append({
                    "pre_img_path": pre_img_path,
                    "post_img_path": post_img_path,
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
                    
                    # Find all post-disaster JSON files (which contain damage labels)
                    post_label_files = [f for f in os.listdir(labels_dir) if f.endswith("_post_disaster.json")]
                    print(f"Disaster {disaster}: Found {len(post_label_files)} label files")
                    
                    # Process each label file
                    for post_label_file in post_label_files:
                        base_id = post_label_file.replace("_post_disaster.json", "")
                        post_img_name = base_id + "_post_disaster.png"
                        pre_img_name = base_id + "_pre_disaster.png"
                        pre_json_path = os.path.join(labels_dir, base_id + "_pre_disaster.json")
                        
                        post_json_path = os.path.join(labels_dir, post_label_file)
                        post_img_path = os.path.join(images_dir, post_img_name)
                        pre_img_path = os.path.join(images_dir, pre_img_name)
                        
                        # Skip if files don't exist
                        if not (os.path.isfile(post_json_path) and os.path.isfile(post_img_path) and 
                                os.path.isfile(pre_json_path) and os.path.isfile(pre_img_path)):
                            continue
                        
                        samples.append({
                            "pre_img_path": pre_img_path,
                            "post_img_path": post_img_path,
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
        Returns:
            pre_image_tensor: Pre-disaster image tensor
            post_image_tensor: Post-disaster image tensor
            damage_mask: Multi-class damage mask with 5 classes
        """
        item = self.samples[idx]
        pre_img_path = item["pre_img_path"]
        post_img_path = item["post_img_path"]
        pre_json_path = item["pre_json_path"]
        post_json_path = item["post_json_path"]
        
        try:
            # Load the pre-disaster and post-disaster images
            pre_img = Image.open(pre_img_path).convert("RGB")
            post_img = Image.open(post_img_path).convert("RGB")
            original_size = pre_img.size  # (width, height)
            
            # Create an empty damage mask (0 = background/non-building)
            damage_mask = Image.new("L", original_size, 0)
            draw = ImageDraw.Draw(damage_mask)
            
            # Load building polygons and damage classes
            with open(pre_json_path, 'r') as f:
                pre_json_data = json.load(f)
                
            with open(post_json_path, 'r') as f:
                post_json_data = json.load(f)
            
            # Map building IDs to damage classes
            building_damage = {}
            post_feats = post_json_data.get("features", {}).get(self.coord_key, [])
            for feat in post_feats:
                uid = feat.get("properties", {}).get("uid", None)
                damage_type = feat.get("properties", {}).get("subtype", "").lower()
                if uid and damage_type in DAMAGE_CLASS_MAP:
                    building_damage[uid] = DAMAGE_CLASS_MAP[damage_type]
            
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
            
            # Apply augmentation if enabled (with the same transformations to both images)
            if self.augment:
                # Same random seed for both transforms to ensure consistency
                seed = np.random.randint(2147483647)
                random.seed(seed)
                torch.manual_seed(seed)
                
                # Simple rotation/flip augmentation
                if random.random() > 0.5:
                    pre_img = T.functional.hflip(pre_img)
                    post_img = T.functional.hflip(post_img)
                    damage_mask = T.functional.hflip(damage_mask)
                if random.random() > 0.5:
                    pre_img = T.functional.vflip(pre_img)
                    post_img = T.functional.vflip(post_img)
                    damage_mask = T.functional.vflip(damage_mask)
                if random.random() > 0.5:
                    angle = random.choice([90, 180, 270])
                    pre_img = T.functional.rotate(pre_img, angle)
                    post_img = T.functional.rotate(post_img, angle)
                    damage_mask = T.functional.rotate(damage_mask, angle)
                
                # Color augmentation to pre-disaster image only
                if random.random() > 0.5:
                    brightness = random.uniform(0.8, 1.2)
                    contrast = random.uniform(0.8, 1.2)
                    saturation = random.uniform(0.8, 1.2)
                    pre_img = T.functional.adjust_brightness(pre_img, brightness)
                    pre_img = T.functional.adjust_contrast(pre_img, contrast)
                    pre_img = T.functional.adjust_saturation(pre_img, saturation)
            
            # Apply transforms to convert images to tensors
            pre_tensor = self.image_transform(pre_img)
            post_tensor = self.image_transform(post_img)
            
            # Resize the damage mask and convert to tensor
            damage_mask = damage_mask.resize((self.image_size, self.image_size), Image.NEAREST)
            damage_tensor = torch.from_numpy(np.array(damage_mask)).long()
            
            return pre_tensor, post_tensor, damage_tensor
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return placeholder tensors in case of error
            pre_tensor = torch.zeros(3, self.image_size, self.image_size)
            post_tensor = torch.zeros(3, self.image_size, self.image_size)
            damage_tensor = torch.zeros(self.image_size, self.image_size, dtype=torch.long)
            return pre_tensor, post_tensor, damage_tensor


def visualize_improved_predictions(model, dataset, device, num_samples=4, save_path=None):
    """
    Visualize improved damage classification predictions
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
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = [axes]  # Make sure axes is a list for single sample case
    
    # Track per-class IoU
    class_iou = {cls: [] for cls in range(5)}
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            try:
                # Get sample
                pre_img, post_img, damage_mask = dataset[idx]
                pre_img = pre_img.unsqueeze(0).to(device)
                post_img = post_img.unsqueeze(0).to(device)
                
                # Get prediction
                outputs = model(pre_img, post_img)
                
                # Get predicted class for each pixel
                _, pred_classes = torch.max(outputs, dim=1)
                
                # Move tensors to CPU for visualization
                pred_classes = pred_classes.squeeze().cpu().numpy()
                damage_mask = damage_mask.cpu().numpy()
                
                # Create colored visualizations
                colored_pred = np.zeros((pred_classes.shape[0], pred_classes.shape[1], 3), dtype=np.uint8)
                colored_mask = np.zeros((damage_mask.shape[0], damage_mask.shape[1], 3), dtype=np.uint8)
                
                # Color each class in the prediction and ground truth
                for class_idx in range(len(damage_colors)):
                    colored_pred[pred_classes == class_idx] = damage_colors[class_idx]
                    colored_mask[damage_mask == class_idx] = damage_colors[class_idx]
                
                # Calculate per-class IoU for this sample
                for cls in range(5):
                    true_mask = (damage_mask == cls)
                    pred_mask = (pred_classes == cls)
                    
                    intersection = np.logical_and(true_mask, pred_mask).sum()
                    union = np.logical_or(true_mask, pred_mask).sum()
                    iou = intersection / union if union > 0 else float('nan')
                    
                    if not np.isnan(iou):
                        class_iou[cls].append(iou)
                
                # Denormalize images for display
                pre_img_np = pre_img.squeeze().cpu().numpy()
                post_img_np = post_img.squeeze().cpu().numpy()
                
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                
                pre_img_np = np.transpose(pre_img_np, (1, 2, 0))
                pre_img_np = pre_img_np * std + mean
                pre_img_np = np.clip(pre_img_np, 0, 1)
                
                post_img_np = np.transpose(post_img_np, (1, 2, 0))
                post_img_np = post_img_np * std + mean
                post_img_np = np.clip(post_img_np, 0, 1)
                
                # Plot in the current row
                axes[i][0].imshow(pre_img_np)
                axes[i][0].set_title("Pre-disaster Image")
                axes[i][0].axis('off')
                
                axes[i][1].imshow(post_img_np)
                axes[i][1].set_title("Post-disaster Image")
                axes[i][1].axis('off')
                
                axes[i][2].imshow(colored_mask)
                axes[i][2].set_title("Ground Truth Damage")
                axes[i][2].axis('off')
                
                axes[i][3].imshow(colored_pred)
                axes[i][3].set_title("Predicted Damage")
                axes[i][3].axis('off')
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                # In case of error, fill the row with blank plots
                for j in range(4):
                    axes[i][j].imshow(np.zeros((10, 10, 3)))
                    axes[i][j].set_title("Error")
                    axes[i][j].axis('off')
    
    # Calculate average IoU for each class
    avg_class_iou = {}
    for cls in range(5):
        if class_iou[cls]:
            avg_class_iou[cls] = sum(class_iou[cls]) / len(class_iou[cls])
        else:
            avg_class_iou[cls] = float('nan')
    
    # Calculate mean IoU (excluding background)
    valid_ious = [iou for cls, iou in avg_class_iou.items() if cls > 0 and not np.isnan(iou)]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0
    
    # Add a text box with IoU results
    plt.figtext(0.5, 0.01, f"Mean IoU (excluding background): {mean_iou:.4f}", ha="center", 
                fontsize=14, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    text = ""
    for cls in range(5):
        iou_value = avg_class_iou[cls]
        iou_str = f"{iou_value:.4f}" if not np.isnan(iou_value) else "N/A"
        text += f"{class_names[cls]}: {iou_str}   "
    
    plt.figtext(0.5, 0.005, text, ha="center", fontsize=12)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    # Print IoU values
    print("\nPer-Class IoU:")
    for cls in range(5):
        iou_value = avg_class_iou[cls]
        iou_str = f"{iou_value:.4f}" if not np.isnan(iou_value) else "N/A"
        print(f"  {class_names[cls]}: {iou_str}")
    print(f"Mean IoU (excluding background): {mean_iou:.4f}")
    
    plt.close()
    
    return avg_class_iou, mean_iou


def main():
    """
    Main function to run the improved damage classification training pipeline.
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
    
    # Get project root directory
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    # Hyperparameters & settings
    root_dir = os.path.join(project_root, "data", "xBD")
    batch_size = 4
    lr = 0.0002
    num_epochs = 20
    val_ratio = 0.2
    image_size = 256
    num_classes = 5  # Background + 4 damage classes

    # Create output directory
    output_dir = os.path.join(project_root, "output", "dam_classifier")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create versioned run directory
    run_dir, run_num = create_versioned_directory(output_dir, prefix="classifier_run")
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
    
    # Initialize dataset with both pre and post disaster images
    print("Initializing ImprovedDamageDataset for training...")
    full_dataset = ImprovedDamageDataset(
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
    
    # Calculate class frequencies for weighting
    print("Calculating class frequencies for weighting...")
    class_counts = torch.zeros(num_classes)
    
    # Keep track of damage class presence in each sample for weighted sampling
    sample_weights = torch.ones(len(train_indices))
    
    for i, idx in enumerate(tqdm(train_indices, desc="Counting class pixels")):
        _, _, damage_mask = full_dataset[idx]
        
        # Count pixels of each class
        for c in range(num_classes):
            class_count = (damage_mask == c).sum().item()
            class_counts[c] += class_count
            
            # Give higher weights to samples containing damaged buildings
            if c >= 2 and class_count > 0:  # If sample has damage
                damage_weight = (c / num_classes) * 20  # Higher damage classes get higher weights
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
    
    # Apply stronger weighting to rare damage classes
    inverse_freq_weights[2:] *= 2.0  # Double the weights for damage classes
    
    # Cap weights to prevent extremely high values
    inverse_freq_weights = torch.clamp(inverse_freq_weights, min=0.1, max=20.0)
    
    # Move to device
    class_weights = inverse_freq_weights.to(device)
    
    print(f"Calculated class weights: {class_weights.tolist()}")
    
    # Create a weighted sampler for training data to balance class representation
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_indices),
        replacement=True
    )
    
    print("Using weighted sampler to balance training batches")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use weighted sampling
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
    
    # Initialize improved model that uses both pre and post disaster images
    model = ImprovedDamageClassifier(in_channels=3, out_channels=num_classes).to(device)
    
    # Use Focal Loss with stronger focus on hard examples
    gamma = 2.5  # Higher value gives more focus to hard examples
    criterion = FocalLoss(gamma=gamma, alpha=class_weights)
    
    # Log the loss function configuration
    print(f"Using Focal Loss with gamma={gamma} and class weights={class_weights.tolist()}")
    
    # Optimizer with weight decay for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
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
        for pre_imgs, post_imgs, damage_masks in train_pbar:
            pre_imgs = pre_imgs.to(device)
            post_imgs = post_imgs.to(device)
            damage_masks = damage_masks.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass - outputs shape: [batch_size, num_classes, H, W]
            outputs = model(pre_imgs, post_imgs)
            
            # Compute loss using Focal Loss
            loss = criterion(outputs, damage_masks)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()
            
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
            for pre_imgs, post_imgs, damage_masks in val_pbar:
                pre_imgs = pre_imgs.to(device)
                post_imgs = post_imgs.to(device)
                damage_masks = damage_masks.to(device)
                
                # Forward pass
                outputs = model(pre_imgs, post_imgs)
                
                # Compute loss
                loss = criterion(outputs, damage_masks)
                val_loss += loss.item()
                
                # Get predicted class for each pixel
                _, preds = torch.max(outputs, dim=1)
                
                # Calculate IoU for multi-class segmentation
                batch_iou = calculate_iou(preds, damage_masks, threshold=0.5, num_classes=num_classes)
                val_iou += batch_iou
                
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{batch_iou:.4f}"})
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        val_losses.append(avg_val_loss)
        val_ious.append(avg_val_iou)
        epochs_list.append(epoch + 1)
        
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
            visualize_improved_predictions(model, full_dataset, device, num_samples=4, save_path=vis_save_path)
    
    # Copy the best model to the main directory
    if best_model_path and os.path.exists(best_model_path):
        improved_best_path = os.path.join(output_dir, "improved_damage_best.pt")
        shutil.copy2(best_model_path, improved_best_path)
        print(f"Best model copied to {improved_best_path}")
    
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