# Core ML libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

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

# Standard Libraries
import argparse
import logging
import sys
from typing import List, Tuple, Optional
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

class FaceExtractor:
    def __init__(self, device='cuda', config=None): # Accept config
        self.device = device
        # Use config for thresholds if provided, else use defaults
        self.min_confidence = config.get('face_min_confidence', 0.9) if config else 0.9
        self.min_face_size = config.get('face_min_size', 50) if config else 50
        self.logger = get_logger(self.__class__.__name__) # Use logger

        self.logger.info(f"Initializing MTCNN face detector on device: {self.device}")
        try:
            self.mtcnn = MTCNN(margin=14, keep_all=True, factor=0.7, device=device)
        except Exception as e:
            self.logger.error(f"Failed to initialize MTCNN: {e}", exc_info=True)
            raise
        # No temp_dir needed if process_dataset writes directly

    def extract_faces_from_image(self, image_path: str) -> List[Image.Image]:
        """Extract faces from image using configured thresholds."""
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
        """Extract faces from video using configured thresholds and frame count."""
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
            if actual_n_frames <= 0: # Handle case where n_frames might be 0 or negative
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
        """Process dataset using configured parameters."""
        self.logger.info(f"Processing dataset in '{data_dir}', outputting faces to '{output_dir}'")
        self.logger.info(f"Params: Min Confidence={self.min_confidence}, Min Size={self.min_face_size}, Frames/Video={frames_per_video}")
        os.makedirs(output_dir, exist_ok=True)
        processed_paths = []
        labels = []

        for label, subdir in enumerate(['real', 'fake']): # 0: real, 1: fake
            input_subdir = os.path.join(data_dir, subdir)
            output_subdir = os.path.join(output_dir, subdir)
            os.makedirs(output_subdir, exist_ok=True)
            self.logger.info(f"Processing '{subdir}' directory...")

            if not os.path.isdir(input_subdir):
                self.logger.warning(f"Subdirectory not found: {input_subdir}. Skipping.")
                continue

            files = [f for f in os.listdir(input_subdir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov'))] # Add more video formats if needed

            for filename in tqdm(files, desc=f'Processing {subdir}', unit='file'):
                input_path = os.path.join(input_subdir, filename)
                base_name = os.path.splitext(filename)[0]

                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    faces = self.extract_faces_from_image(input_path)
                else:
                    faces = self.extract_faces_from_video(input_path, frames_per_video)

                for i, face in enumerate(faces):
                    try:
                        output_filename = f"{base_name}_face_{i}.jpg" # Save as JPG
                        output_path = os.path.join(output_subdir, output_filename)
                        face.save(output_path, "JPEG")
                        processed_paths.append(output_path)
                        labels.append(label)
                    except Exception as save_err:
                         self.logger.error(f"Failed to save extracted face {i} from {filename}: {save_err}", exc_info=True)

        self.logger.info(f"Dataset processing complete. Extracted {len(processed_paths)} faces.")
        return processed_paths, labels

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
            # Check if path is valid before opening
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found at index {idx}: {image_path}")
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # Log error with more context and return a placeholder
            self.logger.error(f"Error loading item at index {idx}, path '{image_path}': {e}", exc_info=True)
            # Return placeholder - Ensure shape matches transformed output
            # Assuming transform outputs [C, H, W] like [3, 260, 260]
            placeholder_img = torch.zeros((3, self.transform.transforms[0].size[0], self.transform.transforms[0].size[1])) if self.transform else torch.zeros((3, 224, 224))
            placeholder_label = torch.tensor(-1, dtype=torch.long) # Use -1 to indicate error? Or keep 0?
            return placeholder_img, placeholder_label # Must return tensors of expected type

class DeepfakeTrainer:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.config['save_dir'], f'run_{self.timestamp}')
        os.makedirs(self.run_dir, exist_ok=True)
        self.logger.info(f"Run directory created: {self.run_dir}")

        # Setup logging specifically for this run
        setup_logging(log_dir=self.config['save_dir'], run_timestamp=self.timestamp)

        self.model = self._build_model()
        lr = config['learning_rate']
        classifier_lr_mult = config.get('classifier_lr_mult', 5.0) # e.g., Classifier LR 5x base LR
        unfrozen_lr_mult = config.get('unfrozen_lr_mult', 2.0)     # e.g., Unfrozen blocks LR 2x base LR

        classifier_params = list(self.model._fc.parameters())
        unfrozen_block_params = []
        unfreeze_blocks = self.config.get('unfreeze_blocks', 3)
        total_blocks = len(self.model._blocks)
        if unfreeze_blocks > 0:
            for i in range(total_blocks - unfreeze_blocks, total_blocks):
                unfrozen_block_params.extend(list(self.model._blocks[i].parameters()))
            # Include conv_head and bn1
            unfrozen_block_params.extend(list(self.model._conv_head.parameters()))
            unfrozen_block_params.extend(list(self.model._bn1.parameters()))

        # Base parameters (frozen or earlier unfrozen layers)
        base_params = [p for p in self.model.parameters() if p.requires_grad and
                    id(p) not in [id(cp) for cp in classifier_params] and
                    id(p) not in [id(up) for up in unfrozen_block_params]]

        optimizer_grouped_parameters = [
            {'params': base_params, 'lr': lr},
            {'params': unfrozen_block_params, 'lr': lr * unfrozen_lr_mult},
            {'params': classifier_params, 'lr': lr * classifier_lr_mult}
        ]

        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr) # Use AdamW maybe
        self.logger.info(f"Optimizer AdamW set up with differential LRs:")
        self.logger.info(f"  Base LR: {lr}")
        self.logger.info(f"  Unfrozen Blocks LR: {lr * unfrozen_lr_mult}")
        self.logger.info(f"  Classifier LR: {lr * classifier_lr_mult}")

        # --- Initialize Scheduler ---
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=self.config.get('scheduler_patience', 5), verbose=True
        )

        # Early stopping parameters
        self.early_stopping_patience = self.config.get('early_stopping_patience', 10)
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf') # Track best val loss for early stopping

        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'best_val_loss': float('inf')}

        # Create transforms based on config
        self.train_transform = self._get_transforms(augment=True)
        self.val_transform = self._get_transforms(augment=False)

    def _build_model(self):
        """Builds the EfficientNet model."""
        self.logger.info(f"Building model: {self.config['model_version']} with image size {self.config['image_size']}")
        try:
            model = EfficientNet.from_pretrained(self.config['model_version'], num_classes=2)
            # --- Unfreeze Layers ---
            unfreeze_blocks = self.config.get('unfreeze_blocks', 3) # Configurable: How many blocks from the end to unfreeze
            total_blocks = len(model._blocks)
            if unfreeze_blocks > 0:
                self.logger.info(f"Unfreezing the last {unfreeze_blocks} blocks of EfficientNet.")
                # Unfreeze classifier layer first (always trainable)
                for param in model._fc.parameters():
                    param.requires_grad = True
                # Unfreeze specified number of blocks from the end
                for i in range(total_blocks - unfreeze_blocks, total_blocks):
                    self.logger.debug(f"Unfreezing Block {i}")
                    for param in model._blocks[i].parameters():
                        param.requires_grad = True
                # Make sure the final conv head and batch norm are also trainable
                for param in model._conv_head.parameters():
                    param.requires_grad = True
                for param in model._bn1.parameters():
                    param.requires_grad = True
            else:
                self.logger.info("Fine-tuning only the final classifier layer.")
                # Freeze everything except the final layer
                for param in model.parameters():
                    param.requires_grad = False
                for param in model._fc.parameters():
                    param.requires_grad = True
            # --- End Unfreeze ---
            model = model.to(self.device)
            return model
        except Exception as e:
            self.logger.error(f"Failed to build model: {e}", exc_info=True)
            raise

    def _get_transforms(self, augment=False):
        """Gets appropriate transforms for train or validation."""
        img_size = self.config['image_size']
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if augment:
            self.logger.debug("Creating training transforms.")
            # Use CenterCrop *after* Resize if aspect ratio is not square
            transform_list = [
                transforms.Resize(img_size), # Resize shorter edge to img_size, maintains aspect ratio
                transforms.CenterCrop(img_size), # Crop center to get square img_size * img_size
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))
                    ], p=0.3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation((-7,7)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                # Add more mild augmentations if needed
                transforms.ToTensor(),
                normalize,
            ]
        else:
            self.logger.debug("Creating validation transforms.")
            transform_list = [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize,
            ]
        return transforms.Compose(transform_list)

    def prepare_data(self):
        """Prepare dataset: extract faces, split, create dataloaders."""
        self.logger.info("Preparing data...")
        face_extractor = FaceExtractor(self.device, self.config) # Pass config
        processed_dir = os.path.join(self.config['data_dir'], 'processed_faces')

        if self.config.get('clean_start', True) and os.path.exists(processed_dir):
            self.logger.info(f"Removing old processed faces from {processed_dir}")
            shutil.rmtree(processed_dir)

        image_paths, labels = face_extractor.process_dataset(
            self.config['data_dir'],
            processed_dir,
            frames_per_video=self.config['frames_per_video'] # Pass frames count
        )

        if not image_paths:
            self.logger.critical("No faces were extracted. Cannot proceed with training.")
            raise ValueError("Face extraction yielded no results.")

        real_count = sum(1 for label in labels if label == 0)
        fake_count = sum(1 for label in labels if label == 1)
        total_count = real_count + fake_count
        if total_count == 0:
            self.logger.critical("Face extraction yielded no valid labels. Cannot proceed.")
            raise ValueError("No labeled faces found.")

        self.logger.info(f"Initial class distribution: Real={real_count} ({real_count/total_count:.1%}), Fake={fake_count} ({fake_count/total_count:.1%})")

        # Split data
        self.logger.info(f"Splitting data with validation size {self.config['val_split']:.1%}")
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels,
            test_size=self.config['val_split'],
            random_state=self.config['seed'],
            stratify=labels
        )
        self.logger.info(f"Train samples: {len(train_paths)}, Validation samples: {len(val_paths)}")

        # Create datasets
        train_dataset = DeepfakeDataset(train_paths, train_labels, transform=self.train_transform)
        val_dataset = DeepfakeDataset(val_paths, val_labels, transform=self.val_transform)

        # Create dataloaders (use balanced sampler for training)
        self.logger.info("Creating DataLoaders (using WeightedRandomSampler for training)...")
        self.train_loader = self._create_balanced_dataloader(train_dataset, train_labels)
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.config['batch_size'], shuffle=False,
            num_workers=self.config['num_workers'], pin_memory=True
        )

        self._monitor_class_distribution(self.train_loader, 'Training Loader')
        self._monitor_class_distribution(self.val_loader, 'Validation Loader')
        self.logger.info("Data preparation complete.")

    def _create_balanced_dataloader(self, dataset, labels):
        """Creates a DataLoader with weighted random sampling."""
        try:
            class_counts = np.bincount(labels)
            if len(class_counts) < 2:
                self.logger.warning("Only one class found in training labels. Sampler will not balance.")
                return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True,
                                  num_workers=self.config['num_workers'], pin_memory=True, drop_last=True)

            # Weight calculation: inverse frequency
            class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
            sample_weights = class_weights[labels] # Assign weight to each sample

            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

            dataloader = DataLoader(
                dataset, batch_size=self.config['batch_size'], sampler=sampler,
                num_workers=self.config['num_workers'], pin_memory=True, drop_last=True # drop_last can be useful with samplers
            )
            return dataloader
        except Exception as e:
             self.logger.error(f"Error creating balanced dataloader: {e}. Falling back to standard shuffle.", exc_info=True)
             # Fallback to standard shuffling if sampler creation fails
             return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True,
                               num_workers=self.config['num_workers'], pin_memory=True, drop_last=True)

    def _monitor_class_distribution(self, loader, name='Loader'):
        """Monitors class distribution within a DataLoader."""
        counts = {}
        total = 0
        for _, batch_labels in loader:
            for label_val in batch_labels.numpy():
                counts[label_val] = counts.get(label_val, 0) + 1
                total += 1
        if total == 0:
            self.logger.warning(f"Cannot monitor distribution for {name}: Loader is empty.")
            return

        self.logger.info(f"Class distribution in {name} (approx first few batches if large):")
        for label, count in sorted(counts.items()):
            self.logger.info(f"  Class {label}: {count} ({count/total:.1%})")

    def train_epoch(self):
        """Trains the model for one epoch."""
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}/{self.config["epochs"]} [Train]', unit='batch')

        for inputs, labels in progress_bar:
            # Skip batches with error labels (if dataset returns -1)
            valid_indices = labels != -1
            if not valid_indices.any(): continue # Skip if whole batch is invalid
            inputs, labels = inputs[valid_indices], labels[valid_indices]

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            # Optional: Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0) # Weighted average loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{100.*correct/total:.2f}%")

        epoch_loss = total_loss / total if total > 0 else 0
        epoch_acc = correct / total if total > 0 else 0
        return epoch_loss, epoch_acc
        
    def validate(self):
        """Validates the model."""
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        progress_bar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch}/{self.config["epochs"]} [Val]', unit='batch')

        with torch.no_grad():
            for inputs, labels in progress_bar:
                 # Skip batches with error labels
                valid_indices = labels != -1
                if not valid_indices.any(): continue
                inputs, labels = inputs[valid_indices], labels[valid_indices]

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                progress_bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{100.*correct/total:.2f}%")

        epoch_loss = total_loss / total if total > 0 else 0
        epoch_acc = correct / total if total > 0 else 0
        return epoch_loss, epoch_acc

    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.config['epochs']} epochs...")
        start_time = time.time()

        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch + 1 # For logging in train/val loops

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            self.logger.info(f"Epoch {self.current_epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}% | Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}%")

            # LR Scheduling
            self.scheduler.step(val_loss)

            # Checkpointing and Early Stopping
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.logger.info(f"Validation loss improved ({self.best_val_loss:.4f} --> {val_loss:.4f}). Saving best model...")
                self.best_val_loss = val_loss
                self.history['best_val_loss'] = self.best_val_loss # Update history too
                self.save_model('best_model.pth')
                self.early_stopping_counter = 0 # Reset counter
            else:
                self.early_stopping_counter += 1
                self.logger.info(f"Validation loss did not improve. Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                if self.early_stopping_counter >= self.early_stopping_patience:
                    self.logger.info("Early stopping triggered.")
                    break # Stop training

        # --- Post-Training ---
        total_time = time.time() - start_time
        self.logger.info(f"Training finished in {total_time//60:.0f}m {total_time%60:.0f}s")

        self.save_model('final_model.pth')
        self.save_history()
        self.plot_history()

        # Load best model for final evaluation
        self.logger.info("Loading best model for final evaluation...")
        best_model_path = os.path.join(self.run_dir, f'best_model_{self.timestamp}.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Best model loaded.")
            optimal_threshold, roc_auc = self.evaluate_model() # Evaluate the best model
            # Generate plots for the best model's predictions
            self.generate_evaluation_plots(optimal_threshold)
        else:
            self.logger.warning("Best model checkpoint not found. Evaluating final model instead.")
            optimal_threshold, roc_auc = self.evaluate_model() # Evaluate the final model
            self.generate_evaluation_plots(optimal_threshold) # Plots based on final model

    def save_model(self, filename_base):
        """Saves model checkpoint."""
        filename = f"{filename_base.split('.')[0]}_{self.timestamp}.pth"
        save_path = os.path.join(self.run_dir, filename)
        state = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history # Save full history
        }
        try:
            torch.save(state, save_path)
            self.logger.info(f"Model checkpoint saved to {save_path}")
        except Exception as e:
             self.logger.error(f"Failed to save model {filename}: {e}", exc_info=True)

    def save_history(self):
        """Saves training history dict to JSON."""
        history_path = os.path.join(self.run_dir, f'training_history_{self.timestamp}.json')
        try:
            with open(history_path, 'w') as f:
                # Convert potential numpy types in history to native Python types
                serializable_history = {}
                for key, value_list in self.history.items():
                    if isinstance(value_list, list):
                         serializable_history[key] = [float(v) if isinstance(v, (np.float32, np.float64)) else v for v in value_list]
                    else: # Handle single values like best_val_loss
                         serializable_history[key] = float(value_list) if isinstance(value_list, (np.float32, np.float64)) else value_list

                json.dump(serializable_history, f, indent=4)
            self.logger.info(f"Training history saved to {history_path}")
        except Exception as e:
             self.logger.error(f"Failed to save history: {e}", exc_info=True)

    def plot_history(self):
        """Plots training and validation loss and accuracy."""
        self.logger.info("Plotting training history...")
        try:
            epochs = range(1, len(self.history['train_loss']) + 1)
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(epochs, self.history['train_loss'], 'bo-', label='Training loss')
            plt.plot(epochs, self.history['val_loss'], 'ro-', label='Validation loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(epochs, [a * 100 for a in self.history['train_acc']], 'bo-', label='Training acc')
            plt.plot(epochs, [a * 100 for a in self.history['val_acc']], 'ro-', label='Validation acc')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plot_path = os.path.join(self.run_dir, f'training_plot_{self.timestamp}.png')
            plt.savefig(plot_path)
            plt.close() # Close the plot to free memory
            self.logger.info(f"Training plot saved to {plot_path}")
        except Exception as e:
             self.logger.error(f"Failed to plot history: {e}", exc_info=True)

    def evaluate_model(self):
        """Evaluates model on validation set, calculates optimal threshold and AUC."""
        self.logger.info("Evaluating model on validation set...")
        self.model.eval()
        all_probs_fake = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Evaluating', unit='batch'):
                valid_indices = labels != -1
                if not valid_indices.any(): continue
                inputs, labels = inputs[valid_indices], labels[valid_indices]

                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)[:, 1] # Probability of class 1 (Fake)

                all_probs_fake.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_labels = np.array(all_labels)
        all_probs_fake = np.array(all_probs_fake)

        if len(np.unique(all_labels)) < 2:
             self.logger.warning("Evaluation dataset contains only one class. Cannot compute ROC/AUC or optimal threshold.")
             # Save default threshold
             with open("optimal_threshold.json", "w") as f: json.dump({"optimal_threshold": 0.5}, f)
             return 0.5, 0.0 # Default values

        fpr, tpr, thresholds = roc_curve(all_labels, all_probs_fake)
        roc_auc = auc(fpr, tpr)
        self.logger.info(f"Validation ROC AUC: {roc_auc:.4f}")

        # Find optimal threshold (Youden's J)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        # Handle edge case where threshold might be > 1 or < 0 due to floating point issues
        optimal_threshold = max(0.0, min(1.0, float(optimal_threshold)))
        self.logger.info(f"Calculated optimal threshold: {optimal_threshold:.6f}")


        # --- Save Optimal Threshold ---
        threshold_data = {"optimal_threshold": optimal_threshold}
        # Save to current directory (for detector app) and run directory (for record)
        try:
            with open("optimal_threshold.json", "w") as f:
                json.dump(threshold_data, f, indent=4)
            shutil.copy("optimal_threshold.json", os.path.join(self.run_dir, f"optimal_threshold_{self.timestamp}.json"))
            self.logger.info(f"Optimal threshold saved to optimal_threshold.json and {self.run_dir}")
        except Exception as e:
             self.logger.error(f"Failed to save optimal threshold: {e}", exc_info=True)

        # --- Plot ROC ---
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=50, label=f'Optimal Threshold ({optimal_threshold:.3f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            roc_path = os.path.join(self.run_dir, f'roc_curve_{self.timestamp}.png')
            plt.savefig(roc_path)
            plt.close()
            self.logger.info(f"ROC curve saved to {roc_path}")
        except Exception as e:
             self.logger.error(f"Failed to plot ROC curve: {e}", exc_info=True)


        return optimal_threshold, roc_auc

    def generate_evaluation_plots(self, optimal_threshold):
         """Generates Classification Report and Confusion Matrix plots."""
         self.logger.info("Generating evaluation plots...")
         self.model.eval()
         all_preds_optimal = []
         all_labels = []
         with torch.no_grad():
             for inputs, labels in tqdm(self.val_loader, desc='Generating Predictions', unit='batch'):
                 valid_indices = labels != -1
                 if not valid_indices.any(): continue
                 inputs, labels = inputs[valid_indices], labels[valid_indices]

                 inputs = inputs.to(self.device)
                 outputs = self.model(inputs)
                 probabilities_fake = torch.softmax(outputs, dim=1)[:, 1]
                 preds = (probabilities_fake >= optimal_threshold).long() # Apply optimal threshold

                 all_preds_optimal.extend(preds.cpu().numpy())
                 all_labels.extend(labels.cpu().numpy())

         all_labels = np.array(all_labels)
         all_preds_optimal = np.array(all_preds_optimal)

         if len(all_labels) == 0:
              self.logger.warning("No valid labels found during prediction generation for plots.")
              return

         # --- Classification Report ---
         try:
             self.logger.info("\n" + classification_report(all_labels, all_preds_optimal, target_names=['Real', 'Fake']))
             report = classification_report(all_labels, all_preds_optimal, output_dict=True, target_names=['Real', 'Fake'])
             df_report = pd.DataFrame(report).iloc[:-1, :].T # Exclude avg/total row, transpose

             plt.figure(figsize=(8, 4))
             sns.heatmap(df_report[['precision', 'recall', 'f1-score']], annot=True, fmt=".3f", cmap="viridis")
             plt.title(f'Classification Report (Threshold = {optimal_threshold:.3f})')
             report_path = os.path.join(self.run_dir, f'classification_report_{self.timestamp}.png')
             plt.savefig(report_path)
             plt.close()
             self.logger.info(f"Classification report plot saved to {report_path}")
         except Exception as e:
              self.logger.error(f"Failed to generate classification report plot: {e}", exc_info=True)


         # --- Confusion Matrix ---
         try:
             cm = confusion_matrix(all_labels, all_preds_optimal)
             plt.figure(figsize=(6, 5))
             sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                         xticklabels=["Pred Real", "Pred Fake"], yticklabels=["True Real", "True Fake"])
             plt.ylabel('Actual Label')
             plt.xlabel('Predicted Label')
             plt.title(f'Confusion Matrix (Threshold = {optimal_threshold:.3f})')
             cm_path = os.path.join(self.run_dir, f'confusion_matrix_{self.timestamp}.png')
             plt.savefig(cm_path)
             plt.close()
             self.logger.info(f"Confusion matrix plot saved to {cm_path}")
         except Exception as e:
             self.logger.error(f"Failed to generate confusion matrix plot: {e}", exc_info=True)

# --- Main Execution Block ---
if __name__ == "__main__":
    DEFAULT_CONFIG = {
        'data_dir': 'dataset',      
        'save_dir': 'models',       
        'batch_size': 24,           # Adjust based on GPU memory
        'learning_rate': 0.0005,    # 会需要调整
        'epochs': 50,               # Number of training 次数
        'frames_per_video': 25,     # Frames to sample per video
        'clean_start': True,        # Remove old processed faces before starting
        'model_version': 'efficientnet-b2',
        'image_size': 260,          
        'num_workers': 4,           # DataLoader workers （看CPU有多少个core)
        'seed': 42,                 # Random seed
        'val_split': 0.2,           # Validation set proportion
        'face_min_confidence': 0.9, # MTCNN confidence threshold
        'face_min_size': 50,        # Minimum face pixel size
        'scheduler_patience': 5,    # Patience for ReduceLROnPlateau
        'early_stopping_patience': 10, # Patience for early stopping
        'unfreeze_blocks': 3,
        'classifier_lr_mult': 5.0,
        'unfrozen_lr_mult': 2.0,
        'focal_loss_alpha': 0.75,
        'focal_loss_gamma': 2.0
    }

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train Deepfake Detection Model")
    parser.add_argument('--data_dir', type=str, help=f"Path to dataset directory (default: {DEFAULT_CONFIG['data_dir']})")
    parser.add_argument('--save_dir', type=str, help=f"Base directory to save models/results (default: {DEFAULT_CONFIG['save_dir']})")
    parser.add_argument('--epochs', type=int, help=f"Number of training epochs (default: {DEFAULT_CONFIG['epochs']})")
    parser.add_argument('--batch_size', type=int, help=f"Batch size (default: {DEFAULT_CONFIG['batch_size']})")
    parser.add_argument('--lr', type=float, help=f"Learning rate (default: {DEFAULT_CONFIG['learning_rate']})")
    parser.add_argument('--img_size', type=int, help="Image input size (required, e.g., 224 for B0, 260 for B2)")
    parser.add_argument('--model', type=str, help=f"EfficientNet version (default: {DEFAULT_CONFIG['model_version']})")
    parser.add_argument('--workers', type=int, help=f"DataLoader workers (default: {DEFAULT_CONFIG['num_workers']})")
    parser.add_argument('--seed', type=int, help=f"Random seed (default: {DEFAULT_CONFIG['seed']})")
    parser.add_argument('--no_clean', action='store_true', help="Do not remove old processed faces")
    # Add arguments for other config items if needed

    args = parser.parse_args()

    # --- Build Final Configuration ---
    config = DEFAULT_CONFIG.copy()
    # Update from args where provided
    if args.data_dir: config['data_dir'] = args.data_dir
    if args.save_dir: config['save_dir'] = args.save_dir
    if args.epochs: config['epochs'] = args.epochs
    if args.batch_size: config['batch_size'] = args.batch_size
    if args.lr: config['learning_rate'] = args.lr
    if args.model: config['model_version'] = args.model
    if args.workers: config['num_workers'] = args.workers
    if args.seed: config['seed'] = args.seed
    if args.no_clean: config['clean_start'] = False
    # Handle image size based on model if not specified
    if args.img_size:
        config['image_size'] = args.img_size
    else:
        # Auto-detect based on model version (add more mappings as needed)
        model_to_size = {'efficientnet-b0': 224, 'efficientnet-b1': 240, 'efficientnet-b2': 260,
                         'efficientnet-b3': 300, 'efficientnet-b4': 380, 'efficientnet-b5': 456,
                         'efficientnet-b6': 528, 'efficientnet-b7': 600}
        if config['model_version'] in model_to_size:
            config['image_size'] = model_to_size[config['model_version']]
            print(f"Auto-detected image size for {config['model_version']}: {config['image_size']}") # Use print before logging setup
        else:
             # Fallback or raise error if size not specified and model unknown
             print(f"Warning: Image size not specified for {config['model_version']}. Defaulting to {DEFAULT_CONFIG['image_size']}. Use --img_size.")
             config['image_size'] = DEFAULT_CONFIG['image_size'] # Or raise error

    # --- Setup Logging (now uses timestamp from trainer) ---
    # Initial basic config for argument parsing phase
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, handlers=[logging.StreamHandler(sys.stdout)])
    logger = get_logger(__name__)
    logger.info("Initial configuration completed.")
    # Full logging setup happens inside Trainer using its timestamp

    # --- Set Seed ---
    logger.info(f"Setting random seed: {config['seed']}")
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
        # Deterministic algorithms can impact performance, use if needed
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # --- Run Training ---
    trainer = DeepfakeTrainer(config)
    try:
        trainer.prepare_data()
        trainer.train()
        logger.info("Training run completed successfully.")
    except ValueError as ve: # Catch specific expected errors like no faces
         logger.critical(f"Training aborted due to ValueError: {ve}", exc_info=True)
    except FileNotFoundError as fnf:
         logger.critical(f"Training aborted due to FileNotFoundError: {fnf}", exc_info=True)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during training: {e}", exc_info=True)