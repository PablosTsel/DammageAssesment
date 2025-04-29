#!/usr/bin/env python3
# xBD Building Damage Assessment Model
# This script implements a deep learning pipeline for detecting building damage
# from satellite imagery before and after natural disasters
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import random
from datetime import datetime
import time
import torch.nn.functional as F
import json
from PIL import Image
import torchvision.transforms as T
from shapely import wkt
import shutil  # Added for file operations

# Import from localization_model
from scripts.training.localization_model import create_versioned_directory

# Constants and definitions
# Dictionary mapping damage category labels to numerical class indices
DAMAGE_LABELS = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3
}

# Dataset class
class XBDPatchDataset(Dataset):
    """
    On-the-fly dataset that:
      - Scans the dataset directory.
      - In hierarchical mode (flat_structure=False): expects subfolders per disaster (each with an 'images' and 'labels' folder).
      - In flat mode (flat_structure=True): expects root_dir to contain 'images' and 'labels' folders directly.
      - For each post-disaster JSON, enumerates building polygons, computes the bounding box, and center-crops the pre & post images.
      - Returns (pre_patch, post_patch, label) for each building.
    """
    def __init__(self,
                 root_dir,
                 pre_crop_size=128,
                 post_crop_size=224,
                 use_xy=True,
                 max_samples=None,
                 flat_structure=False,
                 augment=False):
        """
        :param root_dir: Directory where the data is stored.
           - If flat_structure is False: root_dir should contain subfolders for each disaster.
           - If True: root_dir should contain 'images' and 'labels' folders directly.
        :param pre_crop_size: Crop size for pre-disaster images.
        :param post_crop_size: Crop size for post-disaster images.
        :param use_xy: Use 'xy' coordinates if True; else use 'lng_lat'.
        :param max_samples: Optional limit on the number of samples.
        :param flat_structure: Whether the folder structure is flat.
        :param augment: If True, applies data augmentation (random horizontal flip and random rotation).
        """
        super().__init__()
        self.root_dir = root_dir
        self.pre_crop_size = pre_crop_size
        self.post_crop_size = post_crop_size
        self.coord_key = "xy" if use_xy else "lng_lat"
        self.max_samples = max_samples
        self.flat_structure = flat_structure
        self.augment = augment

        # Basic transforms - keeping it simple for compatibility
        # Define transforms for pre-disaster images (resize to specified size and normalize)
        self.pre_transform = T.Compose([
            T.Resize((pre_crop_size, pre_crop_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        
        # Define transforms for post-disaster images (resize to specified size and normalize)
        self.post_transform = T.Compose([
            T.Resize((post_crop_size, post_crop_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        # Gather all samples from the dataset directory structure
        self.samples = self._gather_samples()
        
        # Limit the number of samples if specified
        if self.max_samples is not None and len(self.samples) > self.max_samples:
            self.samples = random.sample(self.samples, self.max_samples)

    def _gather_samples(self):
        """
        Parse the dataset directory to find all valid sample pairs (pre/post disaster images with labels).
        Supports both flat and hierarchical directory structures.
        Returns a list of dictionaries with paths to images and corresponding labels.
        """
        samples = []
        # First try to check if "test" directory has a flat structure
        images_dir = os.path.join(self.root_dir, "images")
        labels_dir = os.path.join(self.root_dir, "labels")
        
        if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
            # This looks like a flat structure
            print(f"Detected flat structure at {self.root_dir}")
            label_files = [f for f in os.listdir(labels_dir) if f.endswith("_post_disaster.json")]
            
            for label_file in label_files:
                # Extract base ID from filename (removing the _post_disaster.json suffix)
                base_id = label_file.replace("_post_disaster.json", "")
                post_img_name = base_id + "_post_disaster.png"
                pre_img_name = base_id + "_pre_disaster.png"
                post_json_path = os.path.join(labels_dir, label_file)
                post_img_path = os.path.join(images_dir, post_img_name)
                pre_json_path = os.path.join(labels_dir, base_id + "_pre_disaster.json")
                pre_img_path = os.path.join(images_dir, pre_img_name)
                
                # Skip if any of the required files don't exist
                if not (os.path.isfile(post_json_path) and os.path.isfile(post_img_path)
                        and os.path.isfile(pre_json_path) and os.path.isfile(pre_img_path)):
                    continue
                    
                # Load the post-disaster JSON data which contains damage labels
                with open(post_json_path, 'r') as f:
                    post_data = json.load(f)
                    
                # Extract feature information from the JSON data
                feats = post_data.get("features", {}).get(self.coord_key, [])
                for feat in feats:
                    # Get the damage type (subtype) and convert to lowercase
                    damage_type = feat.get("properties", {}).get("subtype", "").lower()
                    if damage_type not in DAMAGE_LABELS:
                        continue
                        
                    # Convert damage type to numerical label
                    label = DAMAGE_LABELS[damage_type]
                    wkt_str = feat.get("wkt", None)
                    if wkt_str is None:
                        continue
                        
                    # Parse the WKT string to get a polygon object and extract its bounding box
                    polygon = wkt.loads(wkt_str)
                    minx, miny, maxx, maxy = polygon.bounds
                    samples.append({
                        "pre_img": pre_img_path,
                        "post_img": post_img_path,
                        "bbox": (minx, miny, maxx, maxy),
                        "label": label
                    })
        else:
            # Hierarchical structure: root_dir contains subfolders per disaster.
            try:
                # List all directories in the root path (excluding 'spacenet_gt' folder)
                disasters = [d for d in os.listdir(self.root_dir)
                            if os.path.isdir(os.path.join(self.root_dir, d))
                            and d.lower() != "spacenet_gt"]
                
                print(f"Found {len(disasters)} disaster folders")
                
                # Process each disaster folder
                for disaster in disasters:
                    disaster_dir = os.path.join(self.root_dir, disaster)
                    images_dir = os.path.join(disaster_dir, "images")
                    labels_dir = os.path.join(disaster_dir, "labels")
                    
                    # Skip if images or labels directory is missing
                    if not (os.path.isdir(images_dir) and os.path.isdir(labels_dir)):
                        print(f"Warning: Missing images or labels directory for disaster: {disaster}")
                        continue
                        
                    # Find all post-disaster JSON files (which contain damage labels)
                    label_files = [f for f in os.listdir(labels_dir) if f.endswith("_post_disaster.json")]
                    print(f"Disaster {disaster}: Found {len(label_files)} label files")
                    
                    # Process each label file
                    for label_file in label_files:
                        base_id = label_file.replace("_post_disaster.json", "")
                        post_img_name = base_id + "_post_disaster.png"
                        pre_img_name = base_id + "_pre_disaster.png"
                        post_json_path = os.path.join(labels_dir, label_file)
                        post_img_path = os.path.join(images_dir, post_img_name)
                        pre_json_path = os.path.join(labels_dir, base_id + "_pre_disaster.json")
                        pre_img_path = os.path.join(images_dir, pre_img_name)
                        
                        # Skip if any required files are missing
                        if not (os.path.isfile(post_json_path) and os.path.isfile(post_img_path)
                                and os.path.isfile(pre_json_path) and os.path.isfile(pre_img_path)):
                            continue
                            
                        # Load the post-disaster JSON data
                        with open(post_json_path, 'r') as f:
                            post_data = json.load(f)
                            
                        # Process each feature in the JSON data
                        feats = post_data.get("features", {}).get(self.coord_key, [])
                        for feat in feats:
                            damage_type = feat.get("properties", {}).get("subtype", "").lower()
                            if damage_type not in DAMAGE_LABELS:
                                continue
                                
                            label = DAMAGE_LABELS[damage_type]
                            wkt_str = feat.get("wkt", None)
                            if wkt_str is None:
                                continue
                                
                            # Parse the WKT string to get the polygon and extract its bounding box
                            polygon = wkt.loads(wkt_str)
                            minx, miny, maxx, maxy = polygon.bounds
                            samples.append({
                                "pre_img": pre_img_path,
                                "post_img": post_img_path,
                                "bbox": (minx, miny, maxx, maxy),
                                "label": label,
                                "disaster": disaster  # Track which disaster this is from
                            })
            except Exception as e:
                print(f"Error gathering samples: {e}")
                
        print(f"Total gathered samples: {len(samples)}")
        # Check class distribution
        labels = [s["label"] for s in samples]
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("Class distribution:")
        for label, count in zip(unique_labels, counts):
            class_name = [name for name, idx in DAMAGE_LABELS.items() if idx == label][0]
            print(f"  {class_name}: {count} samples ({count/len(samples)*100:.2f}%)")
            
        return samples

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        Returns pre-disaster image, post-disaster image, and damage label.
        """
        item = self.samples[idx]
        pre_path = item["pre_img"]
        post_path = item["post_img"]
        (minx, miny, maxx, maxy) = item["bbox"]
        label = item["label"]

        try:
            # Load the pre and post disaster images
            pre_img = Image.open(pre_path).convert("RGB")
            post_img = Image.open(post_path).convert("RGB")

            # Crop the images around the building bounding box
            pre_crop = self._center_crop(pre_img, minx, miny, maxx, maxy, self.pre_crop_size)
            post_crop = self._center_crop(post_img, minx, miny, maxx, maxy, self.post_crop_size)

            # Use consistent random state for both images if augmentation is enabled
            # This ensures the same random transformations are applied to both pre and post images
            if self.augment:
                seed = np.random.randint(2147483647)
                random.seed(seed)
                torch.manual_seed(seed)
                
            # Apply transformations to pre-disaster image
            pre_tensor = self.pre_transform(pre_crop)
            
            if self.augment:
                # Reset the seed for the second transform to ensure same transformation
                random.seed(seed)
                torch.manual_seed(seed)
                
            # Apply transformations to post-disaster image
            post_tensor = self.post_transform(post_crop)

            return pre_tensor, post_tensor, label
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return a placeholder in case of error (zero tensors and the original label)
            placeholder_pre = torch.zeros(3, self.pre_crop_size, self.pre_crop_size)
            placeholder_post = torch.zeros(3, self.post_crop_size, self.post_crop_size)
            return placeholder_pre, placeholder_post, label

    def _center_crop(self, pil_img, minx, miny, maxx, maxy, crop_size):
        """
        Crop the image around the center of the given bounding box with specified crop size.
        Ensures the crop is within the image boundaries.
        """
        width, height = pil_img.size
        bb_width = maxx - minx
        bb_height = maxy - miny
        # Find the center of the bounding box
        cx = minx + bb_width / 2.0
        cy = miny + bb_height / 2.0
        
        # Calculate crop boundaries ensuring they're within image dimensions
        half = crop_size / 2.0
        left = max(0, min(cx - half, width - crop_size))
        top = max(0, min(cy - half, height - crop_size))
        right = left + crop_size
        bottom = top + crop_size
        return pil_img.crop((left, top, right, bottom))

# Model definition
class AttentionFusion(nn.Module):
    """
    Attention-based feature fusion module that combines features from pre- and post-disaster images.
    Uses a learned attention mechanism to focus on the most relevant features for damage assessment.
    """
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
        # Attention mechanism implemented as a small convolutional network
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()  # Outputs values between 0-1 for attention weights
        )
        
    def forward(self, pre_feat, post_feat):
        """
        Forward pass of the attention fusion module.
        Args:
            pre_feat: Features from pre-disaster image
            post_feat: Features from post-disaster image
        Returns:
            Fused feature map combining information from both inputs
        """
        # Make sure the spatial dimensions match before concatenating
        if pre_feat.shape[2:] != post_feat.shape[2:]:
            # Resize post_feat to match pre_feat's spatial dimensions
            post_feat = F.interpolate(post_feat, size=pre_feat.shape[2:], 
                                      mode='bilinear', align_corners=False)
            
        # Concatenate features along channel dimension
        concat_feat = torch.cat([pre_feat, post_feat], dim=1)
        # Generate attention weights
        attn_weights = self.attention(concat_feat)
        # Apply attention to post features
        weighted_post = post_feat * attn_weights
        # Combine pre and weighted post features
        fused_feat = pre_feat + weighted_post
        return fused_feat

class BaselineModel(nn.Module):
    """
    Siamese-style network that processes pre- and post-disaster images separately,
    then fuses their features using an attention mechanism.
    Uses ResNet50 as the backbone for feature extraction.
    """
    def __init__(self, num_classes=4, pretrained=True, dropout_rate=0.5):
        super(BaselineModel, self).__init__()
        
        # Use a more powerful backbone (ResNet50)
        try:
            # For torch 1.13+
            base_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='IMAGENET1K_V2' if pretrained else None)
        except:
            try:
                # For torch 1.13+
                import torchvision.models as models
                base_model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            except TypeError:
                # For older torch versions
                import torchvision.models as models
                base_model = models.resnet50(pretrained=pretrained)
        
        # Pre-disaster branch - uses layers up to the final FC layer of ResNet50
        self.pre_branch = nn.Sequential(*list(base_model.children())[:-2])
        
        # Post-disaster branch (same architecture but separate weights)
        try:
            # For torch 1.13+
            post_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='IMAGENET1K_V2' if pretrained else None)
        except:
            try:
                # For torch 1.13+
                import torchvision.models as models
                post_model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            except TypeError:
                # For older torch versions
                import torchvision.models as models
                post_model = models.resnet50(pretrained=pretrained)
        
        self.post_branch = nn.Sequential(*list(post_model.children())[:-2])
        
        # Get the number of features from the backbone
        self.feature_dim = base_model.fc.in_features  # 2048 for ResNet50
        
        # Attention fusion module to combine features from both branches
        self.attention_fusion = AttentionFusion(self.feature_dim)
        
        # Global average pooling to reduce spatial dimensions
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # MLP head with dropout for better generalization
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights of the classifier
        self._initialize_weights(self.classifier)
        
    def _initialize_weights(self, module):
        """Initialize weights of the model layers using Kaiming initialization."""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, pre_images, post_images):
        """
        Forward pass of the model.
        Args:
            pre_images: Batch of pre-disaster images
            post_images: Batch of post-disaster images
        Returns:
            Classification logits for the damage categories
        """
        # Extract features from pre-disaster images
        pre_features = self.pre_branch(pre_images)
        
        # Extract features from post-disaster images
        post_features = self.post_branch(post_images)
        
        # Fuse features with attention mechanism
        fused_features = self.attention_fusion(pre_features, post_features)
        
        # Global average pooling
        pooled_features = self.gap(fused_features).view(-1, self.feature_dim)
        
        # Classification
        output = self.classifier(pooled_features)
        
        return output

# Functions for training
def seed_everything(seed=42):
    """
    Set random seeds for reproducibility across all libraries and components.
    This helps ensure consistent results between runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

def compute_sample_weights(dataset, indices, weight_scale=1.0):
    """
    Compute sample weights for the given dataset indices to address class imbalance.
    
    Args:
        dataset: The dataset containing samples
        indices: Indices of the samples to use for weight calculation
        weight_scale: Controls the strength of class balancing (higher values give more weight to minority classes)
    
    Returns:
        sample_weights: Weights for each sample
        class_weights: Weights for each class
    """
    # Extract labels for the given indices
    labels = [dataset.samples[i]["label"] for i in indices]
    counts = Counter(labels)
    total = sum(counts.values())
    num_classes = len(DAMAGE_LABELS)
    
    # Compute inverse frequency weights
    # Gives higher weights to under-represented classes
    class_weights = {cls: (total / (num_classes * counts[cls]))**weight_scale for cls in counts}
    
    # Create sample weights
    sample_weights = [class_weights[label] for label in labels]
    
    return sample_weights, class_weights

def mixup_data(x1, x2, y, alpha=0.2, device='cuda'):
    """
    Applies mixup augmentation to the data.
    
    Mixup creates new training examples by linearly interpolating between two random inputs
    and their corresponding labels.
    
    Args:
        x1, x2: Pre and post disaster input tensors
        y: Target labels
        alpha: Parameter for Beta distribution
        device: Device to use
    
    Returns:
        Mixed inputs, original labels, and mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x1, mixed_x2, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Criterion for mixup training - combines losses using the same mixing coefficient.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a, y_b: Original and permuted labels
        lam: Mixing coefficient
    
    Returns:
        Weighted average of losses for the two labels
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def plot_learning_curves(epochs, train_losses, val_losses, val_accuracies, per_class_f1, class_names, save_path):
    """
    Plot and save training metrics visualizations.
    
    Args:
        epochs: List of epoch numbers
        train_losses: Training loss values for each epoch
        val_losses: Validation loss values for each epoch
        val_accuracies: Validation accuracy values for each epoch
        per_class_f1: F1 scores for each class across epochs
        class_names: Names of damage classes
        save_path: File path to save the plot
    """
    plt.figure(figsize=(16, 12))
    
    # Plot loss curves
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker='o', color='blue')
    plt.plot(epochs, val_losses, label="Val Loss", marker='o', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_accuracies, label="Val Accuracy (%)", marker='o', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot per-class F1 scores
    plt.subplot(2, 2, 3)
    per_class_f1 = np.array(per_class_f1)  # shape: (num_epochs, num_classes)
    for i, cls_name in enumerate(class_names):
        plt.plot(epochs, per_class_f1[:, i], label=f"{cls_name}", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Per-Class F1 Scores")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot class distribution of last batch
    plt.subplot(2, 2, 4)
    plt.bar(class_names, [per_class_f1[-1, i] for i in range(len(class_names))])
    plt.xlabel("Class")
    plt.ylabel("Final F1 Score")
    plt.title("Final F1 Score by Class")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Learning curves plot saved to {save_path}")

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss down-weights easy examples (high confidence predictions) and focuses
    training on hard examples (low confidence), helping with class imbalance.
    
    Args:
        alpha: Optional weight for each class
        gamma: Focusing parameter (higher values increase focus on hard examples)
        reduction: How to reduce the loss ('mean', 'sum', or None)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Weight for each class
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Use nn.functional directly instead of F
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def create_versioned_directory(base_path, prefix="trainingTry"):
    """
    Create a versioned directory with incremented try number if base exists.
    Helps keep track of multiple training runs without overwriting previous results.
    
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

def main():
    """
    Main function to run the training pipeline.
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
    batch_size = 32
    lr = 0.0001
    num_epochs = 20
    val_ratio = 0.15
    use_focal_loss = True
    use_mixup = False  # Disable mixup for now
    weight_scale = 0.7
    
    # Create versioned output directories
    base_output_dir = os.path.join(project_root, "output")
    os.makedirs(base_output_dir, exist_ok=True)
    # Create classification subdirectory
    classification_dir = os.path.join(base_output_dir, "ALONE_classifier")
    os.makedirs(classification_dir, exist_ok=True)
    output_dir, try_num = create_versioned_directory(classification_dir)
    
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    pictures_dir = os.path.join(output_dir, "pictures")
    os.makedirs(pictures_dir, exist_ok=True)
    
    # Save configuration
    config = {
        "timestamp": timestamp,
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_epochs": num_epochs,
        "val_ratio": val_ratio,
        "use_focal_loss": use_focal_loss,
        "use_mixup": use_mixup,
        "weight_scale": weight_scale,
    }
    
    with open(os.path.join(output_dir, f"config_try{try_num}.txt"), "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    # Set device (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize dataset
    print("Initializing XBDPatchDataset for training...")
    full_dataset = XBDPatchDataset(
        root_dir=root_dir,
        pre_crop_size=128,
        post_crop_size=224,
        use_xy=True,
        max_samples=None
    )
    total_samples = len(full_dataset)
    print(f"Total samples: {total_samples}")

    # Create stratified train-val split directly
    def create_stratified_split(dataset, val_ratio=0.15, seed=42):
        """
        Create a stratified split of the data ensuring class balance between train and validation sets.
        Each class will have the same proportion of samples in both train and validation sets.
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Group samples by label
        label_to_indices = {}
        for idx, sample in enumerate(dataset.samples):
            label = sample['label']
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        
        train_indices = []
        val_indices = []
        
        # Stratified sampling - take val_ratio proportion from each class
        for label, indices in label_to_indices.items():
            random.shuffle(indices)
            val_size = int(len(indices) * val_ratio)
            val_indices.extend(indices[:val_size])
            train_indices.extend(indices[val_size:])
        
        # Shuffle the indices to avoid any ordering bias
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        
        return train_indices, val_indices
    
    # Create the split
    train_indices, val_indices = create_stratified_split(full_dataset, val_ratio, seed=42)
    print(f"Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}")

    # Compute class weights for weighted sampling
    sample_weights, class_weights = compute_sample_weights(full_dataset, train_indices, weight_scale)
    sampler = SubsetRandomSampler(train_indices)
    
    # Create train and validation datasets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # Compute sample weights for weighted sampling
    # This helps address class imbalance during training
    sample_weights, class_weights = compute_sample_weights(full_dataset, train_indices, weight_scale)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,  # Use weighted sampler for balanced class distribution
        num_workers=16,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,  # No need for sampler with validation set
        num_workers=16,
        pin_memory=True
    )

    # Initialize model - check if pretrained is a valid parameter
    try:
        model = BaselineModel(num_classes=4, pretrained=True, dropout_rate=0.5).to(device)
    except TypeError:
        # If pretrained is not a valid parameter, try without it
        print("Warning: 'pretrained' parameter not supported, using default initialization")
        model = BaselineModel(num_classes=4).to(device)
    
    # Define class weights for loss function
    weight_tensor = torch.tensor([class_weights.get(i, 1.0) for i in range(4)], dtype=torch.float).to(device)
    
    # Loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=weight_tensor, gamma=2.0, reduction='mean')
        print("Using Focal Loss with gamma=2.0")
    else:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print("Using Weighted CrossEntropyLoss")
    
    # Optimizer with weight decay
    # AdamW provides better regularization than standard Adam
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    # OneCycleLR gradually increases then decreases learning rate
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # Percentage of training to increase LR
        anneal_strategy='cos'  # Use cosine annealing
    )

    # Training metrics tracking
    epochs_list = []
    train_losses = []
    val_losses = []
    val_accuracies = []
    per_class_f1_scores = []
    best_f1_score = 0.0
    best_model_path = None  # Track the path of the best model
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        
        # Training phase
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]", leave=True)
        for pre_batch, post_batch, labels in train_pbar:
            pre_batch = pre_batch.to(device)
            post_batch = post_batch.to(device)
            labels = labels.to(device)
            
            # Apply mixup if enabled
            if use_mixup and np.random.random() < 0.5:
                pre_batch, post_batch, labels_a, labels_b, lam = mixup_data(pre_batch, post_batch, labels, alpha=0.2, device=device)
                mixup_applied = True
            else:
                mixup_applied = False

            # Zero gradients before forward pass
            optimizer.zero_grad()
            # Forward pass
            outputs = model(pre_batch, post_batch)
            
            # Compute loss based on whether mixup was applied
            if mixup_applied:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
                
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss_epoch = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [VAL]", leave=True)
            for pre_batch, post_batch, labels in val_pbar:
                pre_batch = pre_batch.to(device)
                post_batch = post_batch.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(pre_batch, post_batch)
                loss = criterion(outputs, labels)
                val_loss_epoch += loss.item()
                
                # Get predictions and calculate accuracy
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                # Collect predictions and labels for F1 score calculation
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
        # Calculate average validation loss
        val_loss_epoch /= len(val_loader)
        val_losses.append(val_loss_epoch)
        val_acc = 100 * correct / total if total > 0 else 0
        val_accuracies.append(val_acc)
        
        # Calculate F1 scores for each class
        epoch_f1 = f1_score(all_labels, all_preds, average=None, labels=[0, 1, 2, 3])
        per_class_f1_scores.append(epoch_f1)
        
        # Calculate macro F1 for model saving
        macro_f1 = np.mean(epoch_f1)
        epochs_list.append(epoch + 1)
        
        # Calculate elapsed time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss_epoch:.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Macro F1: {macro_f1:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        print(f"Per-class F1 Scores: {', '.join([f'{c}: {f:.4f}' for c, f in zip(DAMAGE_LABELS.keys(), epoch_f1)])}")
        
        # Save the model if it's the best so far, and delete previous best
        if macro_f1 > best_f1_score:
            # Delete previous best model file if it exists
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
                print(f"Removed previous best model: {best_model_path}")
                
            best_f1_score = macro_f1
            best_model_path = os.path.join(model_dir, f"best_model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with Macro F1: {best_f1_score:.4f}")

    # Copy the best model as baseline_best.pt (instead of saving a new copy in the root)
    if best_model_path and os.path.exists(best_model_path):
        # Copy to the models directory in the current training run
        baseline_best_path = os.path.join(model_dir, "baseline_best.pt")
        shutil.copy2(best_model_path, baseline_best_path)
        print(f"Best model copied to {baseline_best_path}")
        
        # Also copy to the main classification directory, similar to how localization model does it
        classification_best_path = os.path.join(classification_dir, "baseline_best.pt")
        shutil.copy2(best_model_path, classification_best_path)
        print(f"Best model also copied to {classification_best_path}")
    
    # Create learning curves plot
    plot_save_path = os.path.join(pictures_dir, "learning_curves.png")
    plot_learning_curves(
        epochs_list, 
        train_losses, 
        val_losses, 
        val_accuracies, 
        per_class_f1_scores,
        list(DAMAGE_LABELS.keys()),
        plot_save_path
    )
    
    # Save all training metrics
    metrics = {
        "epochs": epochs_list,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "per_class_f1_scores": per_class_f1_scores,
    }
    
    metrics_path = os.path.join(output_dir, f"training_metrics_try{try_num}.txt")
    with open(metrics_path, "w") as f:
        for key, values in metrics.items():
            if key == "per_class_f1_scores":
                f.write(f"{key}:\n")
                for epoch_idx, epoch_scores in enumerate(values):
                    f.write(f"  Epoch {epoch_idx+1}: {epoch_scores}\n")
            else:
                f.write(f"{key}: {values}\n")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Best validation Macro F1: {best_f1_score:.4f}")
    print(f"All training artifacts saved to {output_dir}")
    print("Run 'evaluate_baseline.py' to test the model")

if __name__ == "__main__":
    main()