#!/usr/bin/env python3
# Two-Stage Building Damage Assessment Evaluation
# This script evaluates the combined building detection and damage classification models

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
from PIL import Image, ImageDraw
import random
import torchvision.transforms as T
from datetime import datetime
import shutil
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns
from shapely import wkt

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

# Import from our other scripts
from scripts.training.utils import UNet, create_versioned_directory

# Import the improved damage classifier
from scripts.training.train_dam_classifier import (
    ImprovedDamageClassifier, AttentionFusion, ImprovedDamageDataset, DAMAGE_CLASS_MAP
)

class IntegratedDamageAssessment:
    """
    End-to-end model that combines building detection and damage classification
    in a two-stage approach.
    """
    def __init__(
        self,
        building_detector_path,
        damage_classifier_path,
        device=None,
        building_threshold=0.5
    ):
        """
        Initialize the integrated model with pre-trained stages.
        
        Args:
            building_detector_path: Path to the saved building detector model
            damage_classifier_path: Path to the saved damage classifier model
            device: Device to run inference on
            building_threshold: Threshold for building detection
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.building_threshold = building_threshold
        
        # Load building detector - binary segmentation model (output: 1 channel)
        self.building_detector = UNet(in_channels=3, out_channels=1).to(self.device)
        self.building_detector.load_state_dict(
            torch.load(building_detector_path, map_location=self.device)
        )
        self.building_detector.eval()
        
        # Load improved damage classifier (dual-input model)
        print("Loading improved damage classifier model (dual-input)")
        self.damage_classifier = ImprovedDamageClassifier(in_channels=3, out_channels=5).to(self.device)
        self.damage_classifier.load_state_dict(
            torch.load(damage_classifier_path, map_location=self.device)
        )
        
        # We need to store the post-disaster images for the improved model
        self.post_disaster_images = {}
        
        self.damage_classifier.eval()
        print("Integrated Damage Assessment model loaded successfully")
        
    def predict(self, image_tensor, post_image_tensor=None):
        """
        Run the full prediction pipeline on an input image.
        
        Args:
            image_tensor: Pre-disaster image tensor of shape [C, H, W] or [B, C, H, W]
            post_image_tensor: Post-disaster image tensor, required for the improved model
            
        Returns:
            building_mask: Binary building mask
            damage_pred: Multi-class damage prediction (5 classes)
            final_pred: Combined prediction with building detection applied
        """
        with torch.no_grad():
            # Add batch dimension if necessary
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            # Move to device
            image_tensor = image_tensor.to(self.device)
            
            # Stage 1: Building Detection
            building_logits = self.building_detector(image_tensor)
            building_mask = (torch.sigmoid(building_logits) > self.building_threshold).float()
            
            # Stage 2: Damage Classification - requires post-disaster image
            if post_image_tensor is None:
                raise ValueError("Post-disaster image is required for the improved model")
            
            # Add batch dimension and move to device
            if post_image_tensor.dim() == 3:
                post_image_tensor = post_image_tensor.unsqueeze(0)
            post_image_tensor = post_image_tensor.to(self.device)
            
            # Get damage prediction from improved model (dual input)
            damage_logits = self.damage_classifier(image_tensor, post_image_tensor)
            
            # Get class probabilities
            damage_pred = F.softmax(damage_logits, dim=1)
            
            # Combine the stages:
            # 1. Multiply damage predictions by building mask so only buildings get classified
            # 2. Use channel 0 (background) where there are no buildings
            
            # Expand building mask to match damage_pred dimensions
            building_mask_expanded = building_mask.expand(-1, damage_pred.size(1), -1, -1)
            
            # Create a mask for background (inverse of building mask)
            background_mask = 1 - building_mask_expanded[:, 0:1]  # Just the first channel
            background_only = torch.zeros_like(damage_pred)
            background_only[:, 0:1] = background_mask  # Set only background channel
            
            # Apply building mask to damage prediction
            building_damage = damage_pred * building_mask_expanded
            
            # Combine: building_damage + background pixels
            final_pred = building_damage + background_only
            
            # Normalize to ensure valid probability distribution
            final_pred = final_pred / (final_pred.sum(dim=1, keepdim=True) + 1e-6)
            
            return building_mask, damage_pred, final_pred


def evaluate_model(integrated_model, dataset, num_samples=None, output_dir=None):
    """
    Evaluate the integrated model on a test dataset.
    
    Args:
        integrated_model: IntegratedDamageAssessment model
        dataset: Test dataset with ground truth
        num_samples: Number of samples to evaluate (None = all)
        output_dir: Directory to save visualization images
    
    Returns:
        Dictionary of metrics
    """
    device = integrated_model.device
    class_names = ['Background', 'No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
    
    # Define colors for damage classes
    damage_colors = [
        [0, 0, 0],       # Background (black)
        [0, 255, 0],     # No damage (green)
        [255, 255, 0],   # Minor damage (yellow)
        [255, 165, 0],   # Major damage (orange)
        [255, 0, 0]      # Destroyed (red)
    ]
    
    # Track metrics
    class_correct = np.zeros(5)
    class_total = np.zeros(5)
    class_pred_total = np.zeros(5)
    confusion = np.zeros((5, 5), dtype=np.int64)
    
    # Track building detection metrics
    total_building_tp = 0
    total_building_union = 0
    
    # Sample indices
    if num_samples is None or num_samples >= len(dataset):
        indices = list(range(len(dataset)))
        num_samples = len(dataset)
    else:
        indices = random.sample(range(len(dataset)), num_samples)
    
    # Set up visualization directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
    
    # Evaluate all samples
    for idx, sample_idx in enumerate(tqdm(indices, desc="Evaluating samples")):
        # Get sample - the dataset should return (pre_img, post_img, building_mask, damage_mask)
        # or (pre_img, building_mask, damage_mask) for the original dataset
        sample_data = dataset[sample_idx]
        
        # Check if we have post-disaster images (for improved model)
        if len(sample_data) == 4:  # Pre-image, post-image, building-mask, damage-mask
            pre_img, post_img, gt_building_mask, gt_damage_mask = sample_data
        else:  # Original dataset format: pre-image, building-mask, damage-mask
            pre_img, gt_building_mask, gt_damage_mask = sample_data
            post_img = None
        
        # Get predictions
        with torch.no_grad():
            # Run the integrated model
            building_mask, damage_pred, final_pred = integrated_model.predict(pre_img, post_img)
            
            # Get class predictions
            _, pred_classes = torch.max(final_pred, dim=1)
            
            # Move to CPU and numpy for evaluation
            pred_classes = pred_classes.squeeze().cpu().numpy()
            building_mask = building_mask.squeeze().cpu().numpy()
            gt_building_mask = gt_building_mask.squeeze().cpu().numpy()
            gt_damage_mask = gt_damage_mask.cpu().numpy()
            
            # Calculate metrics
            # True positive for buildings
            building_tp = np.logical_and(building_mask > 0.5, gt_building_mask > 0.5).sum()
            # Building IoU
            building_union = np.logical_or(building_mask > 0.5, gt_building_mask > 0.5).sum()
            building_iou = building_tp / building_union if building_union > 0 else 0
            
            # Accumulate building detection metrics
            total_building_tp += building_tp
            total_building_union += building_union
            
            # Update pixel-level confusion matrix
            for y in range(pred_classes.shape[0]):
                for x in range(pred_classes.shape[1]):
                    pred_class = pred_classes[y, x]
                    true_class = gt_damage_mask[y, x]
                    confusion[true_class, pred_class] += 1
            
            # Calculate per-class metrics
            for cls in range(5):
                # True pixels of this class
                true_pixels = (gt_damage_mask == cls)
                # Predicted pixels of this class
                pred_pixels = (pred_classes == cls)
                
                # Update totals
                class_total[cls] += true_pixels.sum()
                class_pred_total[cls] += pred_pixels.sum()
                
                # True positives (correctly classified)
                correct = np.logical_and(true_pixels, pred_pixels).sum()
                class_correct[cls] += correct
        
        # Create visualizations for a few samples
        if output_dir and idx < 10:  # Only visualize the first 10 samples
            # Create colored visualizations
            colored_pred = np.zeros((pred_classes.shape[0], pred_classes.shape[1], 3), dtype=np.uint8)
            colored_gt = np.zeros((gt_damage_mask.shape[0], gt_damage_mask.shape[1], 3), dtype=np.uint8)
            building_colored = np.zeros((building_mask.shape[0], building_mask.shape[1], 3), dtype=np.uint8)
            gt_building_colored = np.zeros((gt_building_mask.shape[0], gt_building_mask.shape[1], 3), dtype=np.uint8)
            
            # Color each class in the prediction and ground truth
            for class_idx in range(len(damage_colors)):
                colored_pred[pred_classes == class_idx] = damage_colors[class_idx]
                colored_gt[gt_damage_mask == class_idx] = damage_colors[class_idx]
            
            # Color building masks
            building_colored[building_mask > 0.5] = [255, 255, 255]
            gt_building_colored[gt_building_mask > 0.5] = [255, 255, 255]
            
            # Prepare image for display
            pre_img_np = pre_img.squeeze().cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            pre_img_np = np.transpose(pre_img_np, (1, 2, 0))
            pre_img_np = pre_img_np * std + mean
            pre_img_np = np.clip(pre_img_np, 0, 1)
            
            # Create figure with appropriate number of rows/columns
            if post_img is not None:
                # Create a larger figure to accommodate post-disaster image
                fig, axes = plt.subplots(2, 4, figsize=(24, 12))
                
                # Also prepare post-disaster image
                post_img_np = post_img.squeeze().cpu().numpy()
                post_img_np = np.transpose(post_img_np, (1, 2, 0))
                post_img_np = post_img_np * std + mean
                post_img_np = np.clip(post_img_np, 0, 1)
                
                # Plot results
                axes[0, 0].imshow(pre_img_np)
                axes[0, 0].set_title("Pre-disaster Image")
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(post_img_np)
                axes[0, 1].set_title("Post-disaster Image")
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(gt_building_colored)
                axes[0, 2].set_title("Ground Truth Buildings")
                axes[0, 2].axis('off')
                
                axes[0, 3].imshow(building_colored)
                axes[0, 3].set_title(f"Predicted Buildings (IoU: {building_iou:.4f})")
                axes[0, 3].axis('off')
                
                axes[1, 0].imshow(pre_img_np)
                axes[1, 0].set_title("Pre-disaster Image")
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(post_img_np)
                axes[1, 1].set_title("Post-disaster Image")
                axes[1, 1].axis('off')
                
                axes[1, 2].imshow(colored_gt)
                axes[1, 2].set_title("Ground Truth Damage")
                axes[1, 2].axis('off')
                
                axes[1, 3].imshow(colored_pred)
                axes[1, 3].set_title("Predicted Damage")
                axes[1, 3].axis('off')
            else:
                # Standard 2x3 figure for original model
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                
                # Plot results
                axes[0, 0].imshow(pre_img_np)
                axes[0, 0].set_title("Pre-disaster Image")
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(gt_building_colored)
                axes[0, 1].set_title("Ground Truth Buildings")
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(building_colored)
                axes[0, 2].set_title(f"Predicted Buildings (IoU: {building_iou:.4f})")
                axes[0, 2].axis('off')
                
                axes[1, 0].imshow(pre_img_np)
                axes[1, 0].set_title("Pre-disaster Image")
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(colored_gt)
                axes[1, 1].set_title("Ground Truth Damage")
                axes[1, 1].axis('off')
                
                axes[1, 2].imshow(colored_pred)
                axes[1, 2].set_title("Predicted Damage")
                axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # Save the visualization
            fig_path = os.path.join(viz_dir, f"sample_{sample_idx}.png")
            plt.savefig(fig_path)
            plt.close()
    
    # Calculate overall metrics
    metrics = {}
    
    # Calculate building detection IoU using the accumulated values
    metrics["building_iou"] = total_building_tp / total_building_union if total_building_union > 0 else 0
    
    # Calculate per-class IoU for damage classification
    class_iou = {}
    for cls in range(5):
        union = class_total[cls] + class_pred_total[cls] - class_correct[cls]
        if class_total[cls] == 0:
            class_iou[cls] = float('nan')  # No ground truth pixels for this class
        else:
            class_iou[cls] = class_correct[cls] / union if union > 0 else 0
    
    # Calculate mean IoU across all damage classes
    valid_ious = [iou for cls, iou in class_iou.items() if not np.isnan(iou)]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0
    
    # Store class IoUs in metrics
    metrics["class_iou"] = class_iou
    metrics["mean_damage_iou"] = mean_iou
    
    # Calculate per-class precision and recall
    precision = {}
    recall = {}
    f1_score = {}
    
    for cls in range(5):
        # Precision: TP / (TP + FP)
        if class_pred_total[cls] == 0:
            precision[cls] = float('nan')  # No predictions for this class
        else:
            precision[cls] = class_correct[cls] / class_pred_total[cls]
        
        # Recall: TP / (TP + FN)
        if class_total[cls] == 0:
            recall[cls] = float('nan')  # No ground truth pixels for this class
        else:
            recall[cls] = class_correct[cls] / class_total[cls]
        
        # F1 Score: 2 * (precision * recall) / (precision + recall)
        if np.isnan(precision[cls]) or np.isnan(recall[cls]) or (precision[cls] + recall[cls] == 0):
            f1_score[cls] = float('nan')
        else:
            f1_score[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls])
    
    # Store metrics
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1_score"] = f1_score
    
    # Create confusion matrix visualization
    if output_dir:
        plt.figure(figsize=(10, 8))
        # Normalize by ground truth (rows)
        confusion_norm = confusion.astype('float') / (confusion.sum(axis=1)[:, np.newaxis] + 1e-6)
        sns.heatmap(
            confusion_norm, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Normalized)')
        conf_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(conf_matrix_path)
        plt.close()
        
        # Save raw confusion matrix to file
        np.save(os.path.join(output_dir, "confusion_matrix.npy"), confusion)
    
    # Print summary metrics
    print("\n--- Two-Stage Model Evaluation Results ---")
    print(f"Number of samples evaluated: {num_samples}")
    print(f"Building Detection IoU: {metrics['building_iou']:.4f}")
    print(f"Mean Damage IoU: {metrics['mean_damage_iou']:.4f}")
    
    print("\nPer-Class Damage Assessment Metrics:")
    for cls in range(5):
        class_name = class_names[cls]
        iou_val = class_iou[cls]
        prec_val = precision[cls]
        rec_val = recall[cls]
        f1_val = f1_score[cls]
        
        iou_str = f"{iou_val:.4f}" if not np.isnan(iou_val) else "N/A"
        prec_str = f"{prec_val:.4f}" if not np.isnan(prec_val) else "N/A"
        rec_str = f"{rec_val:.4f}" if not np.isnan(rec_val) else "N/A"
        f1_str = f"{f1_val:.4f}" if not np.isnan(f1_val) else "N/A"
        
        print(f"  {class_name}: IoU={iou_str}, Precision={prec_str}, Recall={rec_str}, F1={f1_str}")
    
    # If output directory is provided, save results to file
    if output_dir:
        results_path = os.path.join(output_dir, "evaluation_results.txt")
        with open(results_path, "w") as f:
            f.write("Two-Stage Model Evaluation Results\n")
            f.write(f"Number of samples evaluated: {num_samples}\n")
            f.write(f"Building Detection IoU: {metrics['building_iou']:.4f}\n")
            f.write(f"Mean Damage IoU: {metrics['mean_damage_iou']:.4f}\n\n")
            
            f.write("Per-Class Damage Assessment Metrics:\n")
            for cls in range(5):
                class_name = class_names[cls]
                iou_val = class_iou[cls]
                prec_val = precision[cls]
                rec_val = recall[cls]
                f1_val = f1_score[cls]
                
                iou_str = f"{iou_val:.4f}" if not np.isnan(iou_val) else "N/A"
                prec_str = f"{prec_val:.4f}" if not np.isnan(prec_val) else "N/A"
                rec_str = f"{rec_val:.4f}" if not np.isnan(rec_val) else "N/A"
                f1_str = f"{f1_val:.4f}" if not np.isnan(f1_val) else "N/A"
                
                f.write(f"  {class_name}: IoU={iou_str}, Precision={prec_str}, Recall={rec_str}, F1={f1_str}\n")
    
    return metrics


# Create a special dataset class for evaluation with the improved model
class ImprovedEvaluationDataset(Dataset):
    """
    Dataset for evaluating the improved damage classification model.
    Provides both pre and post disaster images along with ground truth masks.
    """
    def __init__(self, 
                 root_dir,
                 building_detector_path,
                 device,
                 image_size=256,
                 use_xy=True,
                 augment=False):
        
        # We need the building detector only to use the same image transformation pipeline
        # but we'll use ground truth masks instead of detector predictions
        self.device = device
        self.image_size = image_size
        self.use_xy = use_xy
        self.augment = augment
        self.root_dir = root_dir
        
        # Load building detector just for reference but we won't use it for prediction
        self.building_detector = UNet(in_channels=3, out_channels=1).to(device)
        self.building_detector.load_state_dict(
            torch.load(building_detector_path, map_location=device)
        )
        self.building_detector.eval()
        
        # Image transforms
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Set coordinate key based on preference
        self.coord_key = "xy" if use_xy else "wkt"
        
        # Gather samples with both pre and post disaster images
        self.samples = self._gather_samples()
        
        print(f"Initialized ImprovedEvaluationDataset with {len(self.samples)} samples")
    
    def _gather_samples(self):
        """
        Parse the dataset directory to find all valid samples with pre and post disaster images
        and corresponding JSON labels.
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
                post_img_name = base_id + "_post_disaster.png"
                post_label_file = base_id + "_post_disaster.json"
                
                pre_json_path = os.path.join(labels_dir, pre_label_file)
                pre_img_path = os.path.join(images_dir, pre_img_name)
                post_img_path = os.path.join(images_dir, post_img_name)
                post_json_path = os.path.join(labels_dir, post_label_file)
                
                # Skip if files don't exist
                if not (os.path.isfile(pre_json_path) and os.path.isfile(pre_img_path) and 
                        os.path.isfile(post_img_path) and os.path.isfile(post_json_path)):
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
                    
                    # Find all pre-disaster JSON files
                    pre_label_files = [f for f in os.listdir(labels_dir) if f.endswith("_pre_disaster.json")]
                    print(f"Disaster {disaster}: Found {len(pre_label_files)} label files")
                    
                    # Process each label file
                    for pre_label_file in pre_label_files:
                        base_id = pre_label_file.replace("_pre_disaster.json", "")
                        pre_img_name = base_id + "_pre_disaster.png"
                        post_img_name = base_id + "_post_disaster.png"
                        post_label_file = base_id + "_post_disaster.json"
                        
                        pre_json_path = os.path.join(labels_dir, pre_label_file)
                        pre_img_path = os.path.join(images_dir, pre_img_name)
                        post_img_path = os.path.join(images_dir, post_img_name)
                        post_json_path = os.path.join(labels_dir, post_label_file)
                        
                        # Skip if files don't exist
                        if not (os.path.isfile(pre_json_path) and os.path.isfile(pre_img_path) and 
                                os.path.isfile(post_img_path) and os.path.isfile(post_json_path)):
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
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset with pre and post disaster images 
        and ground truth masks for building footprints and damage classification.
        
        Returns:
            pre_img: Pre-disaster image tensor
            post_img: Post-disaster image tensor
            building_mask: Ground truth binary building mask
            damage_mask: Ground truth multi-class damage mask
        """
        item = self.samples[idx]
        pre_img_path = item["pre_img_path"]
        post_img_path = item["post_img_path"]
        pre_json_path = item["pre_json_path"]
        post_json_path = item["post_json_path"]
        
        try:
            # Load pre and post disaster images
            pre_img = Image.open(pre_img_path).convert("RGB")
            post_img = Image.open(post_img_path).convert("RGB")
            original_size = pre_img.size  # (width, height)
            
            # Create empty building mask and damage mask
            building_mask = Image.new("L", original_size, 0)
            damage_mask = Image.new("L", original_size, 0)
            building_draw = ImageDraw.Draw(building_mask)
            damage_draw = ImageDraw.Draw(damage_mask)
            
            # First, create a mapping of building UIDs to damage classes from post-disaster JSON
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
                
                # Draw the polygon filled on both masks
                building_draw.polygon(coords, fill=1)  # 1 = building
                damage_draw.polygon(coords, fill=damage_class)  # Fill with damage class
            
            # Apply transforms to images
            pre_img_tensor = self.image_transform(pre_img)
            post_img_tensor = self.image_transform(post_img)
            
            # Resize masks and convert to tensors
            building_mask = building_mask.resize((self.image_size, self.image_size), Image.NEAREST)
            damage_mask = damage_mask.resize((self.image_size, self.image_size), Image.NEAREST)
            
            building_tensor = torch.from_numpy(np.array(building_mask)).float().unsqueeze(0)
            damage_tensor = torch.from_numpy(np.array(damage_mask)).long()
            
            return pre_img_tensor, post_img_tensor, building_tensor, damage_tensor
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return placeholder tensors in case of error
            pre_img_tensor = torch.zeros(3, self.image_size, self.image_size)
            post_img_tensor = torch.zeros(3, self.image_size, self.image_size)
            building_tensor = torch.zeros(1, self.image_size, self.image_size)
            damage_tensor = torch.zeros(self.image_size, self.image_size, dtype=torch.long)
            return pre_img_tensor, post_img_tensor, building_tensor, damage_tensor


def main():
    # Set multiprocessing start method to 'spawn' to avoid CUDA initialization errors
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'")
    except RuntimeError:
        print("Multiprocessing start method already set to 'spawn' or could not be set")
        pass
        
    # Get project paths
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    # Model paths
    building_detector_path = os.path.join(project_root, "output", "building_detector", "binary_building_best.pt")
    
    # Check if improved damage classifier exists first
    improved_damage_path = os.path.join(project_root, "output", "dam_classifier", "improved_damage_best.pt")

    # Only use the improved damage classifier
    if os.path.exists(improved_damage_path):
        damage_classifier_path = improved_damage_path
        print(f"Using improved damage classifier: {improved_damage_path}")
    else:
        print(f"ERROR: Improved damage classifier model not found at: {improved_damage_path}")
        print("Please train the damage classifier using scripts/training/train_damage_classifier.py")
        return

    # Verify that the model files exist
    if not os.path.exists(building_detector_path):
        print(f"ERROR: Building detector model not found at: {building_detector_path}")
        print("Please train the building detector first using scripts/training/train_building_detector.py")
        return
    
    # Create output directory for evaluation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "output", "evaluation_two_stage")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create versioned run directory
    run_dir, run_num = create_versioned_directory(output_dir, prefix="eval_run")
    os.makedirs(run_dir, exist_ok=True)
    
    # Settings
    image_size = 256
    use_xy = True
    test_ratio = 0.2  # Portion of dataset to use for testing
    
    # Dataset path
    root_dir = os.path.join(project_root, "data", "xBD")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the integrated model
    integrated_model = IntegratedDamageAssessment(
        building_detector_path=building_detector_path,
        damage_classifier_path=damage_classifier_path,
        device=device,
        building_threshold=0.5  # Threshold for building detection
    )
    
    # Create a dataset for evaluation
    print("Initializing test dataset...")
    try:
        test_dataset = ImprovedEvaluationDataset(
            root_dir=root_dir,
            building_detector_path=building_detector_path,
            device=device,
            image_size=image_size,
            use_xy=use_xy,
            augment=False  # No augmentation for testing
        )
    except Exception as e:
        print(f"Error initializing ImprovedEvaluationDataset: {e}")
        print("Cannot run evaluation without the improved dataset that provides post-disaster images")
        return
    
    # Determine test samples
    # Choose either a random subset or all samples depending on test_ratio
    dataset_size = len(test_dataset)
    num_test_samples = int(dataset_size * test_ratio)
    
    print(f"Evaluating on {num_test_samples} samples from a dataset of {dataset_size} total samples")
    
    # Run evaluation
    metrics = evaluate_model(
        integrated_model=integrated_model,
        dataset=test_dataset,
        num_samples=num_test_samples,
        output_dir=run_dir
    )
    
    print(f"Evaluation complete. Results saved to: {run_dir}")
    
    # Save the model configuration
    config = {
        "timestamp": timestamp,
        "building_detector_path": building_detector_path,
        "damage_classifier_path": damage_classifier_path,
        "building_threshold": 0.5,
        "image_size": image_size,
        "test_samples": num_test_samples,
        "building_iou": metrics["building_iou"],
        "mean_damage_iou": metrics["mean_damage_iou"]
    }
    
    config_path = os.path.join(run_dir, "config.txt")
    with open(config_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main() 