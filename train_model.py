# Core ML libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import io

# Data processing
from PIL import Image, UnidentifiedImageError
import os
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import cv2
from facenet_pytorch import MTCNN
from sklearn.model_selection import train_test_split

# Visualization and metrics
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score, balanced_accuracy_score

# Standard Libraries
import argparse
import logging
import sys
from typing import List, Tuple
import time

# --- Logger Setup (Copy or Import from logger_setup.py) ---
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO # Change to logging.DEBUG for more verbose logs

def setup_logging(log_dir="logs", run_timestamp=""):
    """Configures logging to file and console for a specific run."""
    log_run_dir = os.path.join(log_dir, f"run_{run_timestamp}")
    os.makedirs(log_run_dir, exist_ok=True)
    log_file = os.path.join(log_run_dir, "training.log")

    # Remove previous handlers if reconfiguring
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file, mode='a'), # Append mode for the run
            logging.StreamHandler(sys.stdout)       # Console output
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")

def get_logger(name):
    """Gets a logger instance."""
    return logging.getLogger(name)

# --- Focal Loss Implementation ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', num_classes=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

        # Handle alpha
        if isinstance(alpha, (float, int)):
            if self.num_classes == 2:
                self.alpha = torch.tensor([1 - alpha, alpha], dtype=torch.float32)
            else:
                self.alpha = torch.tensor([alpha] * self.num_classes, dtype=torch.float32)
        elif isinstance(alpha, (list, tuple, np.ndarray, torch.Tensor)):
            if len(alpha) != self.num_classes:
                raise ValueError(f"Alpha must be a float or a list/tensor of length num_classes ({self.num_classes})")
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            raise TypeError("Alpha must be float, list, tuple, np.ndarray, or torch.Tensor")

    def forward(self, inputs, targets):
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        alpha_t = self.alpha.gather(0, targets.data.view(-1))
        F_loss = alpha_t * (1 - pt)**self.gamma * CE_loss
        if self.reduction == 'mean': return torch.mean(F_loss)
        elif self.reduction == 'sum': return torch.sum(F_loss)
        else: return F_loss

class FaceExtractor:
    def __init__(self, device='cuda', config=None):
        self.device = device
        self.min_confidence = config.get('face_min_confidence', 0.9) if config else 0.9
        self.min_face_size = config.get('face_min_size', 50) if config else 50
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"Initializing MTCNN face detector on device: {self.device}")
        try:
            self.mtcnn = MTCNN(margin=14, keep_all=True, factor=0.7, device=device)
        except Exception as e:
            self.logger.error(f"Failed to initialize MTCNN: {e}", exc_info=True)
            raise

    def extract_faces_from_image(self, image_path: str) -> List[Image.Image]:
        try:
            img = Image.open(image_path).convert('RGB')
            boxes, probs = self.mtcnn.detect(img)
            if boxes is None or probs is None: return []
            extracted_faces = []
            for box, confidence in zip(boxes, probs):
                if confidence < self.min_confidence: continue
                box_int = [int(b) for b in box]
                w, h = box_int[2] - box_int[0], box_int[3] - box_int[1]
                if w < self.min_face_size or h < self.min_face_size: continue
                face = img.crop((box_int[0], box_int[1], box_int[2], box_int[3]))
                extracted_faces.append(face)
            return extracted_faces
        except UnidentifiedImageError:
            self.logger.warning(f"Cannot identify image file (possibly corrupt): {image_path}")
            return []
        except Exception as e:
            self.logger.error(f"Error extracting faces from image '{image_path}': {e}", exc_info=True)
            return []

    def extract_faces_from_video(self, video_path: str, n_frames: int) -> List[Image.Image]:
        extracted_faces = []
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Could not open video file: {video_path}")
                return []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                self.logger.warning(f"Video has 0 or invalid frames: {video_path}")
                return []
            actual_n_frames = min(n_frames, total_frames)
            if actual_n_frames <= 0:
                 self.logger.warning(f"Invalid number of frames to sample ({actual_n_frames}) for video: {video_path}")
                 return []
            frame_indices = np.linspace(0, total_frames - 1, actual_n_frames, dtype=int)
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret: continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                boxes, probs = self.mtcnn.detect(frame_pil)
                if boxes is None or probs is None: continue
                for box, confidence in zip(boxes, probs):
                    if confidence < self.min_confidence: continue
                    box_int = [int(b) for b in box]
                    w, h = box_int[2] - box_int[0], box_int[3] - box_int[1]
                    if w < self.min_face_size or h < self.min_face_size: continue
                    face = frame_pil.crop((box_int[0], box_int[1], box_int[2], box_int[3]))
                    extracted_faces.append(face)
            return extracted_faces
        except Exception as e:
            self.logger.error(f"Error extracting faces from video '{video_path}': {e}", exc_info=True)
            return []
        finally:
            if cap: cap.release()

    def process_dataset(self, data_dir: str, output_dir: str, frames_per_video: int) -> Tuple[List[str], List[int]]:
        self.logger.info(f"Processing dataset in '{data_dir}', outputting faces to '{output_dir}'")
        self.logger.info(f"Params: Min Confidence={self.min_confidence}, Min Size={self.min_face_size}, Frames/Video={frames_per_video}")
        os.makedirs(output_dir, exist_ok=True)
        processed_paths = []
        labels = []
        for label, subdir in enumerate(['real', 'fake']):
            input_subdir = os.path.join(data_dir, subdir)
            output_subdir = os.path.join(output_dir, subdir)
            os.makedirs(output_subdir, exist_ok=True)
            self.logger.info(f"Processing '{subdir}' directory...")
            if not os.path.isdir(input_subdir):
                self.logger.warning(f"Subdirectory not found: {input_subdir}. Skipping.")
                continue
            files = [f for f in os.listdir(input_subdir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov'))]
            for filename in tqdm(files, desc=f'Processing {subdir}', unit='file'):
                input_path = os.path.join(input_subdir, filename)
                base_name = os.path.splitext(filename)[0]
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    faces = self.extract_faces_from_image(input_path)
                else:
                    faces = self.extract_faces_from_video(input_path, frames_per_video)
                for i, face in enumerate(faces):
                    try:
                        output_filename = f"{base_name}_face_{i}.jpg"
                        output_path = os.path.join(output_subdir, output_filename)
                        face.save(output_path, "JPEG")
                        processed_paths.append(output_path)
                        labels.append(label)
                    except Exception as save_err:
                         self.logger.error(f"Failed to save extracted face {i} from {filename}: {save_err}", exc_info=True)
        self.logger.info(f"Dataset processing complete. Extracted {len(processed_paths)} faces.")
        return processed_paths, labels

# Custom Transform for JPEG Compression
class RandomJPEGCompression(object):
    def __init__(self, quality_range=(50, 95), p=0.5): # Slightly wider quality range
        self.quality_min = quality_range[0]
        self.quality_max = quality_range[1]
        self.p = p
        self.logger = get_logger(self.__class__.__name__)

    def __call__(self, img): # img is a PIL Image
        if torch.rand(1) < self.p:
            try:
                quality = torch.randint(self.quality_min, self.quality_max + 1, (1,)).item()
                output = io.BytesIO()
                img.save(output, format="JPEG", quality=quality)
                output.seek(0)
                return Image.open(output).convert('RGB') # Ensure it's RGB after loading
            except Exception as e:
                # self.logger.warning(f"RandomJPEGCompression failed: {e}. Returning original image.")
                return img
        return img

class DeepfakeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.logger = get_logger(self.__class__.__name__)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        try:
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found at index {idx}: {image_path}. Returning placeholder.")
                ph_size = 224 # Default
                if self.transform and hasattr(self.transform, 'transforms'):
                    for t in self.transform.transforms:
                        if isinstance(t, (transforms.Resize, transforms.CenterCrop)):
                            ph_size = t.size if isinstance(t.size, int) else t.size[0]; break
                return torch.zeros((3, ph_size, ph_size)), torch.tensor(-1, dtype=torch.long), image_path
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, image_path
        except Exception as e:
            self.logger.error(f"Error loading item at index {idx}, path '{image_path}': {e}. Returning placeholder.", exc_info=True)
            ph_size = 224 # Default
            if self.transform and hasattr(self.transform, 'transforms'):
                for t in self.transform.transforms:
                    if isinstance(t, (transforms.Resize, transforms.CenterCrop)):
                        ph_size = t.size if isinstance(t.size, int) else t.size[0]; break
            return torch.zeros((3, ph_size, ph_size)), torch.tensor(-1, dtype=torch.long), image_path

class DeepfakeTrainer:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda': self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.config['save_dir'], f'run_{self.timestamp}')
        os.makedirs(self.run_dir, exist_ok=True)
        self.logger.info(f"Run directory created: {self.run_dir}")
        
        # Setup logging specifically for this run AFTER run_dir is created
        setup_logging(log_dir=self.config['save_dir'], run_timestamp=self.timestamp)

        # **** MOVED self.model ASSIGNMENT EARLIER ****
        self.model = self._build_model() # Build the model and assign to self.model

        # Now that self.model exists, you can set up the optimizer
        lr = config['learning_rate']
        classifier_lr_mult = config.get('classifier_lr_mult', 1.0)
        unfrozen_lr_mult = config.get('unfrozen_lr_mult', 1.0)
        
        classifier_params = list(self.model._fc.parameters()) # Now self.model._fc is valid
        unfrozen_block_params = []
        unfreeze_blocks_count = self.config.get('unfreeze_blocks', 0) # Default to 0 for baseline

        if unfreeze_blocks_count > 0:
            total_blocks = len(self.model._blocks)
            for i in range(total_blocks - unfreeze_blocks_count, total_blocks):
                unfrozen_block_params.extend(list(self.model._blocks[i].parameters()))
            if hasattr(self.model, '_conv_head'): unfrozen_block_params.extend(list(self.model._conv_head.parameters()))
            if hasattr(self.model, '_bn1'): unfrozen_block_params.extend(list(self.model._bn1.parameters()))
        base_params = [p for p in self.model.parameters() if p.requires_grad and
                       id(p) not in [id(cp) for cp in classifier_params] and
                       id(p) not in [id(up) for up in unfrozen_block_params]]
        optimizer_grouped_parameters = [
            {'params': base_params, 'lr': lr},
            {'params': unfrozen_block_params, 'lr': lr * unfrozen_lr_mult},
            {'params': classifier_params, 'lr': lr * classifier_lr_mult}
        ]
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=1e-4) # Added weight_decay
        self.logger.info(f"Optimizer AdamW: BaseLR={lr}, ClassifierLRMult={classifier_lr_mult}, UnfrozenLRMult={unfrozen_lr_mult}, WeightDecay=1e-4")
        if self.config.get('use_focal_loss', False): # Check config for FocalLoss
            self.logger.info("Using Focal Loss.")
            self.criterion = FocalLoss(alpha=self.config.get('focal_loss_alpha', 0.25), gamma=self.config.get('focal_loss_gamma', 2.0), num_classes=2)
        else:
            self.logger.info("Using CrossEntropyLoss for baseline.")
            self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=self.config.get('scheduler_patience', 5), verbose=True, min_lr=1e-7) # Added min_lr
        self.early_stopping_patience = self.config.get('early_stopping_patience', 10)
        self.early_stopping_counter = 0; self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'best_val_loss': float('inf')}
        self.train_transform = self._get_transforms(augment=True)
        self.val_transform = self._get_transforms(augment=False)
        self.test_paths = []; self.test_labels = []; self.optimal_threshold_from_val = 0.5

    def _build_model(self):
        self.logger.info(f"Building model: {self.config['model_version']} with image size {self.config['image_size']}")
        try:
            model = EfficientNet.from_pretrained(self.config['model_version'], num_classes=2)
            unfreeze_blocks_count = self.config.get('unfreeze_blocks', 0) # Default to 0 for baseline
            # Freeze all params first
            for param in model.parameters(): param.requires_grad = False
            # Always make classifier trainable
            for param in model._fc.parameters(): param.requires_grad = True
            if unfreeze_blocks_count > 0:
                total_blocks = len(model._blocks)
                self.logger.info(f"Unfreezing last {unfreeze_blocks_count} blocks, _conv_head, _bn1.")
                for i in range(total_blocks - unfreeze_blocks_count, total_blocks):
                    for param in model._blocks[i].parameters(): param.requires_grad = True
                if hasattr(model, '_conv_head'):
                    for param in model._conv_head.parameters(): param.requires_grad = True
                if hasattr(model, '_bn1'):
                    for param in model._bn1.parameters(): param.requires_grad = True
            else: self.logger.info("Fine-tuning only final classifier layer (_fc).")
            return model.to(self.device)
        except Exception as e: self.logger.error(f"Model build error: {e}", exc_info=True); raise

    def _get_transforms(self, augment=False):
        img_size = self.config['image_size']
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if augment:
            self.logger.info("Creating training transforms (CONSERVATIVE BASELINE SET).")
            transform_list = [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01), # Mild
                transforms.ToTensor(),
                normalize,
            ]
        else:
            self.logger.debug("Creating validation/test transforms.")
            transform_list = [transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor(), normalize]
        return transforms.Compose(transform_list)

    def prepare_data(self):
        self.logger.info("Preparing data...")
        face_extractor = FaceExtractor(self.device, self.config)
        processed_dir = os.path.join(self.config['data_dir'], 'processed_faces')
        if self.config.get('clean_start', True) and os.path.exists(processed_dir):
            self.logger.info(f"Removing old processed faces from {processed_dir}"); shutil.rmtree(processed_dir)
        image_paths, labels = face_extractor.process_dataset(self.config['data_dir'], processed_dir, frames_per_video=self.config['frames_per_video'])
        if not image_paths: self.logger.critical("No faces extracted!"); raise ValueError("Face extraction yielded no results.")
        real_count = sum(1 for l in labels if l==0); fake_count = len(labels) - real_count; total_count = len(labels)
        if total_count == 0: self.logger.critical("No labeled faces found!"); raise ValueError("No labeled faces found.")
        self.logger.info(f"Total extracted: {total_count}. Real={real_count} ({real_count/total_count:.1%}), Fake={fake_count} ({fake_count/total_count:.1%})")

        test_split_ratio = self.config.get('test_split', 0.2)
        val_split_ratio_of_remaining = self.config.get('val_split', 0.25)
        if not (0 <= test_split_ratio < 1): self.logger.warning(f"Invalid test_split {test_split_ratio}, using 0.2."); test_split_ratio = 0.2
        if not (0 < val_split_ratio_of_remaining < 1): self.logger.warning(f"Invalid val_split {val_split_ratio_of_remaining}, using 0.25."); val_split_ratio_of_remaining = 0.25

        train_val_paths, self.test_paths, train_val_labels, self.test_labels = image_paths, [], labels, []
        if test_split_ratio > 0:
            if len(np.unique(labels)) > 1:
                train_val_paths, self.test_paths, train_val_labels, self.test_labels = train_test_split(image_paths, labels, test_size=test_split_ratio, random_state=self.config['seed'], stratify=labels)
            else: # Cannot stratify with one class
                self.logger.warning("Only one class in dataset, unstratified test split.")
                train_val_paths, self.test_paths, train_val_labels, self.test_labels = train_test_split(image_paths, labels, test_size=test_split_ratio, random_state=self.config['seed'])
        self.logger.info(f"After test split: Train/Val pool={len(train_val_paths)}, Test={len(self.test_paths)}")

        train_paths, val_paths, train_labels, val_labels = [], [], [], []
        if len(train_val_paths) > 0:
            if len(np.unique(train_val_labels)) > 1:
                train_paths, val_paths, train_labels, val_labels = train_test_split(train_val_paths, train_val_labels, test_size=val_split_ratio_of_remaining, random_state=self.config['seed'], stratify=train_val_labels)
            else: # Cannot stratify with one class
                self.logger.warning("Only one class in train_val pool, unstratified val split.")
                train_paths, val_paths, train_labels, val_labels = train_test_split(train_val_paths, train_val_labels, test_size=val_split_ratio_of_remaining, random_state=self.config['seed'])
        self.logger.info(f"Final sizes: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(self.test_paths)}")

        if not train_paths: self.logger.critical("Training set empty!"); raise ValueError("Training set empty.")
        if not val_paths: self.logger.critical("Validation set empty!"); raise ValueError("Validation set empty.")

        train_dataset = DeepfakeDataset(train_paths, train_labels, transform=self.train_transform)
        val_dataset = DeepfakeDataset(val_paths, val_labels, transform=self.val_transform)
        self.train_loader = self._create_balanced_dataloader(train_dataset, train_labels)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'], pin_memory=True)
        self._monitor_class_distribution(self.train_loader, 'Training Loader (Sampled)')
        self._monitor_class_distribution(self.val_loader, 'Validation Loader (Actual)')
        if self.test_paths:
            test_ds_mon = DeepfakeDataset(self.test_paths, self.test_labels,transform=self.val_transform)
            test_ld_mon = DataLoader(test_ds_mon, batch_size=self.config['batch_size'])
            self._monitor_class_distribution(test_ld_mon, 'Test Loader (Actual)')
        self.logger.info("Data preparation complete.")

    def _create_balanced_dataloader(self, dataset, labels):
        try:
            if not labels: # Handle empty labels list
                 self.logger.warning("Empty labels list provided to _create_balanced_dataloader. Returning standard DataLoader.")
                 return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'], pin_memory=True, drop_last=True)

            class_counts = np.bincount(labels)
            if len(class_counts) < 2:
                self.logger.warning("Only one class found in training labels. Sampler will not balance effectively.")
                return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'], pin_memory=True, drop_last=True)
            class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
            sample_weights = class_weights[labels]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            return DataLoader(dataset, batch_size=self.config['batch_size'], sampler=sampler, num_workers=self.config['num_workers'], pin_memory=True, drop_last=True)
        except Exception as e:
             self.logger.error(f"Error creating balanced dataloader: {e}. Falling back.", exc_info=True)
             return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'], pin_memory=True, drop_last=True)

    def _monitor_class_distribution(self, loader, name='Loader'):
        counts = {}; total_samples_in_loader = 0
        max_samples_to_check = 5000 if 'Train' in name else float('inf')
        try: # Add try-except for robustness with the new third return value from __getitem__
            for _, batch_labels, _ in loader: # Unpack the third item (image_path)
                for label_val in batch_labels.numpy():
                    if label_val == -1: continue
                    counts[label_val] = counts.get(label_val, 0) + 1
                    total_samples_in_loader += 1
                if total_samples_in_loader >= max_samples_to_check and 'Train' in name : break
        except ValueError: # If loader does not return 3 items (e.g. old val_loader)
            self.logger.debug(f"Falling back to 2-item unpacking for {name} distribution check.")
            total_samples_in_loader = 0 # Reset
            counts = {}
            for _, batch_labels in loader:
                for label_val in batch_labels.numpy():
                    if label_val == -1: continue
                    counts[label_val] = counts.get(label_val, 0) + 1
                    total_samples_in_loader += 1
                if total_samples_in_loader >= max_samples_to_check and 'Train' in name : break

        if total_samples_in_loader == 0: self.logger.warning(f"Cannot monitor for {name}: Empty or all error labels."); return
        self.logger.info(f"Class distribution in {name} (~{total_samples_in_loader} samples):")
        for label_val, count in sorted(counts.items()): self.logger.info(f"  Class {label_val}: {count} ({count/total_samples_in_loader:.2%})")

    def train_epoch(self):
        self.model.train(); total_loss, correct, total_samples_processed = 0, 0, 0
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        if self.optimizer.param_groups[0]['params']: self.optimizer.zero_grad()
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}/{self.config["epochs"]} [Train]', unit='batch')
        for i, (inputs, labels, _) in enumerate(progress_bar): # Unpack image_path
            valid_indices = labels != -1
            if not valid_indices.any(): self.logger.debug(f"Skipping batch {i} (train) all error labels."); continue
            inputs, labels = inputs[valid_indices], labels[valid_indices]
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs); loss = self.criterion(outputs, labels)
            loss_scaled = loss / accumulation_steps; loss_scaled.backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                if self.config.get('clip_grad_norm', False):
                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.get('grad_norm_max',1.0))
                if self.optimizer.param_groups[0]['params']: self.optimizer.step(); self.optimizer.zero_grad()
            total_loss += loss.item() * inputs.size(0); _, predicted = torch.max(outputs.data, 1)
            total_samples_processed += labels.size(0); correct += (predicted == labels).sum().item()
            if total_samples_processed > 0:
                progress_bar.set_postfix(loss=f"{total_loss/total_samples_processed:.4f}", acc=f"{100.*correct/total_samples_processed:.2f}%")
        return (total_loss/total_samples_processed if total_samples_processed > 0 else 0), \
               (correct/total_samples_processed if total_samples_processed > 0 else 0)

    def validate(self): # Validation uses self.val_loader
        self.model.eval(); total_loss, correct, total_samples_processed = 0, 0, 0
        progress_bar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch}/{self.config["epochs"]} [Val]', unit='batch')
        with torch.no_grad():
            for inputs, labels, _ in progress_bar: # Unpack image_path
                valid_indices = labels != -1
                if not valid_indices.any(): self.logger.debug("Skipping batch (val) all error labels."); continue
                inputs, labels = inputs[valid_indices], labels[valid_indices]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs); loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0); _, predicted = torch.max(outputs.data, 1)
                total_samples_processed += labels.size(0); correct += (predicted == labels).sum().item()
                if total_samples_processed > 0:
                    progress_bar.set_postfix(loss=f"{total_loss/total_samples_processed:.4f}", acc=f"{100.*correct/total_samples_processed:.2f}%")
        return (total_loss/total_samples_processed if total_samples_processed > 0 else 0), \
               (correct/total_samples_processed if total_samples_processed > 0 else 0)

    def train(self):
        self.logger.info(f"Starting training for {self.config['epochs']} epochs. Criterion: {self.criterion.__class__.__name__}")
        start_time = time.time() # Moved here
        for epoch_num in range(self.config['epochs']):
            self.current_epoch = epoch_num + 1 # current_epoch is 1-based
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.logger.info(f"Epoch {self.current_epoch}: Train L={train_loss:.4f}, A={train_acc*100:.2f}% | Val L={val_loss:.4f}, A={val_acc*100:.2f}%")
            self.scheduler.step(val_loss) # Scheduler step
            if val_loss < self.best_val_loss:
                self.logger.info(f"Val loss improved ({self.best_val_loss:.4f} -> {val_loss:.4f}). Saving best model...")
                self.best_val_loss = val_loss; self.history['best_val_loss'] = self.best_val_loss
                self.save_model('best_model.pth'); self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                self.logger.info(f"Val loss no improvement. Early stop: {self.early_stopping_counter}/{self.early_stopping_patience}")
                if self.early_stopping_counter >= self.early_stopping_patience: self.logger.info("Early stopping."); break
        
        total_training_time = time.time() - start_time
        self.logger.info(f"Training finished in {total_training_time//60:.0f}m {total_training_time%60:.0f}s.")
        self.save_model('final_model.pth'); self.save_history(); self.plot_history()
        
        self.logger.info("Loading best model for VALIDATION set evaluation...")
        best_model_path = os.path.join(self.run_dir, f'best_model_{self.timestamp}.pth')
        if os.path.exists(best_model_path):
            try:
                checkpoint = torch.load(best_model_path, map_location=self.device)
                # Ensure keys match; from_pretrained creates model, so just load state_dict
                model_state_dict = checkpoint.get('model_state_dict', checkpoint) # Handle older checkpoints
                self.model.load_state_dict(model_state_dict); self.logger.info("Best model's state_dict loaded.")
            except Exception as e: self.logger.error(f"Err loading best model: {e}. Eval on final.", exc_info=True)
        else: self.logger.warning(f"Best model not found: {best_model_path}. Eval on final.")

        self.optimal_threshold_from_val, _ = self.evaluate_model(loader_type='validation')
        self.generate_evaluation_plots(self.optimal_threshold_from_val, dataset_type='validation')

        if self.test_paths:
            self.logger.info("Evaluating BEST model on TEST set (using VAL threshold)...")
            self.evaluate_model(loader_type='test', optimal_threshold_override=self.optimal_threshold_from_val)
            self.generate_evaluation_plots(self.optimal_threshold_from_val, dataset_type='test')
        else: self.logger.info("No test set. Skipping test set evaluation.")

    def save_model(self, filename_base):
        filename = f"{filename_base.split('.')[0]}_{self.timestamp}.pth"
        save_path = os.path.join(self.run_dir, filename)
        state = {'epoch': self.current_epoch, 'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.scheduler.state_dict(),
                 'best_val_loss': self.best_val_loss, 'config': self.config, 'history': self.history}
        try: torch.save(state, save_path); self.logger.info(f"Model saved: {save_path}")
        except Exception as e: self.logger.error(f"Failed to save model {filename}: {e}", exc_info=True)

    def save_history(self):
        pth=os.path.join(self.run_dir,f'training_history_{self.timestamp}.json')
        try:
            with open(pth,'w') as f:
                hist_ser={}
                for k,v_list in self.history.items():
                    if isinstance(v_list,list):hist_ser[k]=[float(v) if isinstance(v,(np.float32,np.float64)) else v for v in v_list]
                    else:hist_ser[k]=float(v_list) if isinstance(v_list,(np.float32,np.float64)) else v_list
                json.dump(hist_ser,f,indent=4)
            self.logger.info(f"History saved: {pth}")
        except Exception as e: self.logger.error(f"Save history err: {e}",exc_info=True)

    def plot_history(self):
            self.logger.info("Plotting training history...")
            try:
                if not self.history['train_loss']: # Check if history is empty
                    self.logger.warning("Training history is empty. Skipping plotting.")
                    return

                epochs = range(1, len(self.history['train_loss']) + 1)
                plt.figure(figsize=(14, 6)) # Wider figure

                # Loss Subplot
                plt.subplot(1, 2, 1)
                plt.plot(epochs, self.history['train_loss'], 'bo-', label='Training loss')
                plt.plot(epochs, self.history['val_loss'], 'ro-', label='Validation loss')
                plt.title('Training and Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                # Optional: Set y-limit for loss if needed, e.g., from 0
                min_loss = min(min(self.history['train_loss']), min(self.history['val_loss']))
                max_loss = max(max(self.history['train_loss']), max(self.history['val_loss']))
                plt.ylim(max(0, min_loss - 0.01), max_loss + 0.01) # Ensure y_min is at least 0

                # Accuracy Subplot
                plt.subplot(1, 2, 2)
                train_acc_percent = [a * 100 for a in self.history['train_acc']]
                val_acc_percent = [a * 100 for a in self.history['val_acc']]

                plt.plot(epochs, train_acc_percent, 'bo-', label='Training acc (%)')
                plt.plot(epochs, val_acc_percent, 'ro-', label='Validation acc (%)')
                plt.title('Training and Validation Accuracy')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy (%)')
                plt.legend()
                plt.grid(True)

                if val_acc_percent: # If there are validation accuracies
                    min_val_acc = min(val_acc_percent)
                    y_min_acc = max(0, min_val_acc - 5) # Go 5% below min val_acc, clamped at 0
                    
                    if max(val_acc_percent) < 60 and min_val_acc > 30 : # Example: if accuracies are between 30-60
                        y_min_acc = max(0, min_val_acc - 10) # Give more room below
                    elif min_val_acc < 0: # Should not happen with accuracy
                        y_min_acc = 0

                elif train_acc_percent: # Fallback to train_acc if val_acc is empty
                    min_train_acc = min(train_acc_percent)
                    y_min_acc = max(0, min_train_acc - 5)
                    if min_train_acc < 0: y_min_acc = 0
                else: # If no accuracy data at all
                    y_min_acc = 0

                plt.ylim(y_min_acc, 100.5) # y_max slightly above 100 for clarity

                plt.tight_layout()
                plot_path = os.path.join(self.run_dir, f'training_plot_{self.timestamp}.png')
                plt.savefig(plot_path)
                plt.close() # Close the plot to free memory
                self.logger.info(f"Training plot saved to {plot_path}")
            except Exception as e:
                self.logger.error(f"Plot history err: {e}", exc_info=True)

    def evaluate_model(self, loader_type='validation', optimal_threshold_override=None):
        """Evaluates model, calculates optimal threshold (if validation) and AUC."""
        self.logger.info(f"Evaluating model on {loader_type} set...")
        self.model.eval(); all_probs_fake,all_labels_eval,image_paths_eval = [],[],[]
        dataset_name, loader_to_use = "", None
        if loader_type == 'validation':
            if not hasattr(self, 'val_loader') or not self.val_loader: self.logger.error("Val loader missing!"); return 0.5,0.0
            loader_to_use, dataset_name = self.val_loader, "Validation"
        elif loader_type == 'test':
            if not self.test_paths: self.logger.error("Test paths missing!"); return 0.5,0.0
            test_ds = DeepfakeDataset(self.test_paths, self.test_labels, transform=self.val_transform)
            loader_to_use = DataLoader(test_ds, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'], pin_memory=True)
            dataset_name = "Test"
        else: self.logger.error(f"Unknown eval loader: {loader_type}"); return 0.5,0.0

        with torch.no_grad():
            for inputs, labels_b, paths_b in tqdm(loader_to_use, desc=f'Eval {dataset_name}', unit='b'):
                valid = labels_b != -1
                if not valid.any(): continue
                inputs_f,labels_f = inputs[valid], labels_b[valid]
                paths_f = [p for p,v in zip(paths_b, valid) if v]
                outputs = self.model(inputs_f.to(self.device))
                probs = torch.softmax(outputs, dim=1)[:,1] # Prob of Fake
                all_probs_fake.extend(probs.cpu().numpy()); all_labels_eval.extend(labels_f.cpu().numpy()); image_paths_eval.extend(paths_f)
        
        labels_np, probs_np = np.array(all_labels_eval), np.array(all_probs_fake)
        if len(labels_np) < 2 or len(np.unique(labels_np)) < 2: self.logger.warning(f"{dataset_name} too small/one class. Metrics skip."); return (optimal_threshold_override if optimal_threshold_override is not None else 0.5), 0.0

        fpr,tpr,thresh_roc = roc_curve(labels_np, probs_np); roc_auc = auc(fpr,tpr)
        self.logger.info(f"{dataset_name} ROC AUC: {roc_auc:.4f}")
        
        optimal_threshold = 0.5
        if loader_type == 'validation':
            opt_idx = np.argmax(tpr-fpr); optimal_threshold = float(thresh_roc[opt_idx])
            optimal_threshold = max(0.0, min(1.0, optimal_threshold))
            self.logger.info(f"Optimal VAL threshold (Youden's J): {optimal_threshold:.6f}")
            try: # Save optimal_threshold.json
                with open("optimal_threshold.json","w") as f: json.dump({"optimal_threshold":optimal_threshold},f,indent=4)
                shutil.copy("optimal_threshold.json", os.path.join(self.run_dir, f"optimal_threshold_{self.timestamp}.json"))
                self.logger.info(f"Optimal threshold saved to files.")
            except Exception as e: self.logger.error(f"Save optimal thresh err: {e}", exc_info=True)
        elif optimal_threshold_override is not None:
            optimal_threshold = optimal_threshold_override
            self.logger.info(f"Using VAL threshold for TEST: {optimal_threshold:.6f}")
        else: self.logger.error("TEST eval needs optimal_threshold_override!")

        try: # Save predictions
            df_preds = pd.DataFrame({'true_label':labels_np, 'pred_prob_fake':probs_np, 'image_path':image_paths_eval})
            preds_pth = os.path.join(self.run_dir, f'{dataset_name.lower()}_preds_{self.timestamp}.csv')
            df_preds.to_csv(preds_pth, index=False); self.logger.info(f"{dataset_name} preds saved: {preds_pth}")
        except Exception as e: self.logger.error(f"Save {dataset_name} preds err: {e}", exc_info=True)

        preds_at_opt = (probs_np >= optimal_threshold).astype(int)
        bal_acc = balanced_accuracy_score(labels_np, preds_at_opt)
        self.logger.info(f"{dataset_name} Bal.Acc (thresh {optimal_threshold:.3f}): {bal_acc:.4f}")
        cm = confusion_matrix(labels_np, preds_at_opt)
        if cm.shape == (2,2): tn,fp,fn,tp = cm.ravel(); spec = tn/(tn+fp) if (tn+fp)>0 else 0.0; self.logger.info(f"{dataset_name} Specificity: {spec:.4f}")
        
        prec_pr, rec_pr, _ = precision_recall_curve(labels_np, probs_np)
        avg_prec = average_precision_score(labels_np, probs_np)
        self.logger.info(f"{dataset_name} Avg. Precision (PR AUC): {avg_prec:.4f}")
        
        # Plot PR/ROC (keep existing try-except blocks)
        try: # Plot PR
            plt.figure(figsize=(8,6));plt.plot(rec_pr,prec_pr,color='b',lw=2,label=f'PR (AP={avg_prec:.4f})')
            plt.xlabel('Recall');plt.ylabel('Precision');plt.title(f'{dataset_name} PR Curve');plt.legend(loc='lower left');plt.grid(True)
            pr_pth=os.path.join(self.run_dir,f'{dataset_name.lower()}_pr_curve_{self.timestamp}.png');plt.savefig(pr_pth);plt.close()
            self.logger.info(f"{dataset_name} PR curve saved: {pr_pth}")
        except Exception as e: self.logger.error(f"Plot {dataset_name} PR err: {e}", exc_info=True)
        try: # Plot ROC
            plt.figure(figsize=(8,6));plt.plot(fpr,tpr,color='darkorange',lw=2,label=f'ROC (AUC={roc_auc:.4f})')
            plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
            if loader_type == 'validation': plt.scatter(fpr[np.argmax(tpr-fpr)],tpr[np.argmax(tpr-fpr)],marker='o',color='r',s=50,label=f'OptThresh({optimal_threshold:.3f})')
            plt.xlim([0.0,1.0]);plt.ylim([0.0,1.05]);plt.xlabel('FPR');plt.ylabel('TPR');plt.title(f'{dataset_name} ROC Curve');plt.legend(loc='lower right');plt.grid(True)
            roc_pth=os.path.join(self.run_dir,f'{dataset_name.lower()}_roc_curve_{self.timestamp}.png');plt.savefig(roc_pth);plt.close()
            self.logger.info(f"{dataset_name} ROC curve saved: {roc_pth}")
        except Exception as e: self.logger.error(f"Plot {dataset_name} ROC err: {e}", exc_info=True)

        return optimal_threshold, roc_auc

    def generate_evaluation_plots(self, optimal_threshold, dataset_type='validation'):
        """Generates Classification Report and Confusion Matrix plots for specified dataset type."""
        self.logger.info(f"Generating {dataset_type} eval plots (thresh {optimal_threshold:.4f})...")
        self.model.eval(); all_preds_opt, all_labels_plots = [],[]
        dataset_name, loader_to_use = "", None # Init
        if dataset_type == 'validation':
            if not hasattr(self, 'val_loader') or not self.val_loader: self.logger.error("Val loader missing for plots!"); return
            loader_to_use, dataset_name = self.val_loader, "Validation"
        elif dataset_type == 'test':
            if not self.test_paths: self.logger.error("Test paths missing for plots!"); return
            test_ds = DeepfakeDataset(self.test_paths, self.test_labels, transform=self.val_transform)
            loader_to_use = DataLoader(test_ds, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'], pin_memory=True)
            dataset_name = "Test"
        else: self.logger.error(f"Unknown dataset type for plots: {dataset_type}"); return

        with torch.no_grad():
            for inputs, labels_b, _ in tqdm(loader_to_use, desc=f'Preds for {dataset_name} Plots', unit='b'):
                valid = labels_b != -1
                if not valid.any(): continue
                inputs_f, labels_f = inputs[valid], labels_b[valid]
                outputs = self.model(inputs_f.to(self.device))
                probs_fake = torch.softmax(outputs, dim=1)[:,1]
                preds = (probs_fake >= optimal_threshold).long()
                all_preds_opt.extend(preds.cpu().numpy()); all_labels_plots.extend(labels_f.cpu().numpy())
        
        labels_np, preds_np = np.array(all_labels_plots), np.array(all_preds_opt)
        if len(labels_np) == 0: self.logger.warning(f"No valid labels for {dataset_name} plots."); return

        try: # Classif. Report
            self.logger.info(f"\n{dataset_name} " + classification_report(labels_np, preds_np, target_names=['Real','Fake'], zero_division=0))
            report = classification_report(labels_np, preds_np, output_dict=True, target_names=['Real','Fake'], zero_division=0)
            df_report = pd.DataFrame(report).iloc[:-1,:].T # Exclude avg/total row for heatmap
            plt.figure(figsize=(8,4)); sns.heatmap(df_report[['precision','recall','f1-score']], annot=True,fmt=".3f",cmap="viridis", vmin=0.0, vmax=1.0) # Set vmin/vmax
            plt.title(f'{dataset_name} Classif. Report (Thresh={optimal_threshold:.3f})'); plt.tight_layout()
            rep_pth=os.path.join(self.run_dir,f'{dataset_name.lower()}_classif_report_{self.timestamp}.png'); plt.savefig(rep_pth); plt.close()
            self.logger.info(f"{dataset_name} classif. report plot saved: {rep_pth}")
        except Exception as e: self.logger.error(f"Gen {dataset_name} report plot err: {e}", exc_info=True)
        try: # Confusion Matrix
            cm = confusion_matrix(labels_np, preds_np)
            plt.figure(figsize=(6,5)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred Real","Pred Fake"], yticklabels=["True Real","True Fake"])
            plt.ylabel('Actual Label'); plt.xlabel('Predicted Label'); plt.title(f'{dataset_name} Conf. Matrix (Thresh={optimal_threshold:.3f})'); plt.tight_layout()
            cm_pth=os.path.join(self.run_dir,f'{dataset_name.lower()}_conf_matrix_{self.timestamp}.png'); plt.savefig(cm_pth); plt.close()
            self.logger.info(f"{dataset_name} conf. matrix plot saved: {cm_pth}")
        except Exception as e: self.logger.error(f"Gen {dataset_name} CM plot err: {e}", exc_info=True)

# --- Main Execution Block ---
if __name__ == "__main__":
    DEFAULT_CONFIG = {
        'data_dir': 'dataset',
        'save_dir': 'models',
        'batch_size': 16,
        'learning_rate': 1e-4,
        'epochs': 60,
        'frames_per_video': 50, # Increased frames per video
        'clean_start': True,
        'model_version': 'efficientnet-b0',
        'image_size': 224,
        'num_workers': 4, # Reduced for potentially more I/O with more frames
        'seed': 42,
        'test_split': 0.15,
        'val_split': 0.20,
        'face_min_confidence': 0.85, # Slightly lower min confidence for face extraction
        'face_min_size': 40,       # Slightly smaller min face size
        'scheduler_patience': 10,
        'early_stopping_patience': 20,
        'unfreeze_blocks': 0,       # Start with 0, only classifier
        'classifier_lr_mult': 1.0,  # No multiplication if only classifier
        'unfrozen_lr_mult': 1.0,    # No multiplication if only classifier
        'use_focal_loss': False,    # **Start with CrossEntropyLoss for baseline**
        'focal_loss_alpha': 0.5,
        'focal_loss_gamma': 2.0,
        'gradient_accumulation_steps': 2, # Effective batch_size = 16*2 = 32
        'clip_grad_norm': True,
        'grad_norm_max': 1.0
    }
    # ... (keep existing argparse and config update logic) ...
    parser = argparse.ArgumentParser(description="Train Deepfake Detection Model")
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float, dest='learning_rate')
    parser.add_argument('--img_size', type=int)
    parser.add_argument('--model', type=str, dest='model_version')
    parser.add_argument('--workers', type=int, dest='num_workers')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--frames_per_video', type=int)
    parser.add_argument('--face_min_confidence', type=float)
    parser.add_argument('--face_min_size', type=int)
    parser.add_argument('--test_split', type=float, help="Proportion of data for test set (0.0 to 1.0)")
    parser.add_argument('--val_split', type=float, help="Proportion of remaining (train+val) data for validation set (0.0 to 1.0)")
    parser.add_argument('--no_clean', action='store_false', dest='clean_start')
    parser.add_argument('--use_focal_loss', action='store_true', default=None) # Allow overriding default
    parser.add_argument('--no_focal_loss', action='store_false', dest='use_focal_loss')
    parser.add_argument('--focal_alpha', type=float, dest='focal_loss_alpha')
    parser.add_argument('--unfreeze_blocks', type=int)
    parser.add_argument('--grad_accum_steps', type=int, dest='gradient_accumulation_steps')
    parser.add_argument('--clip_grad', action='store_true', default=None) # Allow overriding default
    parser.add_argument('--no_clip_grad', action='store_false', dest='clip_grad_norm')

    args = parser.parse_args()
    config = DEFAULT_CONFIG.copy()

    for arg_name, arg_val in vars(args).items():
        if arg_val is not None:
            config_key = arg_name
            if arg_name == 'lr': config_key = 'learning_rate'
            elif arg_name == 'model': config_key = 'model_version'
            elif arg_name == 'workers': config_key = 'num_workers'
            elif arg_name == 'grad_accum_steps': config_key = 'gradient_accumulation_steps'
            elif arg_name == 'focal_alpha': config_key = 'focal_loss_alpha'
            config[config_key] = arg_val
    
    if args.img_size is None:
        model_to_size = {'efficientnet-b0': 224, 'efficientnet-b1': 240, 'efficientnet-b2': 260,
                         'efficientnet-b3': 300, 'efficientnet-b4': 380, 'efficientnet-b5': 456,
                         'efficientnet-b6': 528, 'efficientnet-b7': 600}
        if config['model_version'] in model_to_size:
            config['image_size'] = model_to_size[config['model_version']]
            print(f"Auto-set image size for {config['model_version']}: {config['image_size']}")
        else: print(f"Warning: Img size not spec. Using default: {config['image_size']}.")

    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, handlers=[logging.StreamHandler(sys.stdout)])
    logger = get_logger(__name__)
    logger.info("Initial config from args. Final config for trainer:")
    for k,v in config.items(): logger.info(f"  {k}: {v}")
    logger.info(f"Setting random seed: {config['seed']}")
    torch.manual_seed(config['seed']); np.random.seed(config['seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config['seed'])

    trainer = DeepfakeTrainer(config)
    try:
        trainer.prepare_data()
        trainer.train()
        logger.info("Training run completed successfully.")
    except ValueError as ve: logger.critical(f"Training aborted: {ve}", exc_info=True)
    except FileNotFoundError as fnf: logger.critical(f"Training aborted: {fnf}", exc_info=True)
    except Exception as e: logger.critical(f"Unexpected error during training: {e}", exc_info=True)