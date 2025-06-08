import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import sys
import os
import json
import logging
from typing import Tuple, Optional
import threading

logger = logging.getLogger(__name__)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        logger.debug(f"Accessing bundled resource from _MEIPASS: {base_path}")
    except Exception:
        # If not bundled, use the script's directory
        base_path = os.path.abspath(os.path.dirname(__file__))
        logger.debug(f"Accessing resource from script path: {base_path}")

    path = os.path.join(base_path, relative_path)
    logger.debug(f"Resource path requested for '{relative_path}', resolved to: {path}") # Optional: Extra logging
    return path

class DeepfakeDetector:
    def __init__(self, model_version='efficientnet-b2', user_model_dir=None, user_threshold_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.user_model_dir = user_model_dir
        self.user_threshold_path = user_threshold_path
        self.optimal_threshold = 0.5 # Default

        bundled_threshold_path = resource_path("optimal_threshold.json")
        threshold_to_load = None

        if self.user_threshold_path and os.path.exists(self.user_threshold_path):
             logger.info(f"Attempting threshold load from user path: {self.user_threshold_path}")
             threshold_to_load = self.user_threshold_path
        elif os.path.exists(bundled_threshold_path):
             logger.info(f"User threshold not found, using bundled: {bundled_threshold_path}")
             threshold_to_load = bundled_threshold_path
        else:
             logger.warning("No threshold file found (user/bundled). Using default 0.5.")

        if threshold_to_load:
            try:
                with open(threshold_to_load, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    self.optimal_threshold = data.get("optimal_threshold", 0.5)
                logger.info(f"Loaded threshold {self.optimal_threshold:.4f} from {threshold_to_load}")
            except Exception as e:
                logger.error(f"Failed loading threshold from {threshold_to_load}: {e}. Default 0.5.", exc_info=True)
                self.optimal_threshold = 0.5

        search_paths = []
        if self.user_model_dir and os.path.exists(self.user_model_dir):
             search_paths.append(self.user_model_dir)
             logger.info(f"Adding user model dir to search path: {self.user_model_dir}")

        try:
             # Look for the 'default_model' folder bundled via resource_path
             default_model_bundled_dir = resource_path('default_model')
             if os.path.isdir(default_model_bundled_dir):
                  search_paths.append(default_model_bundled_dir)
                  logger.info(f"Adding bundled default model dir to search path: {default_model_bundled_dir}")
             else:
                  logger.info("No bundled default model directory found.")
        except Exception as e:
             logger.warning(f"Error trying to access bundled default model path: {e}")

        model_path = self.get_latest_model_path(search_paths) # Search both user and default

        if not model_path:
            logger.critical("FATAL: No model file found in any search path!")
            raise FileNotFoundError("Deepfake model file not found.")

        logger.info(f"Loading model from: {model_path}")

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.CenterCrop(260),     
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        logger.info(f"Instantiating model: {model_version}")
        self.model = EfficientNet.from_pretrained(model_version, num_classes=2).to(self.device)

        # Load model weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Handle potential variations in checkpoint keys
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            # Adjust for potential DataParallel prefix
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load model state_dict from {model_path}: {e}", exc_info=True)
            raise

    def get_latest_model_path(self, search_paths: list) -> Optional[str]:
        """Finds the latest model file within provided search paths.
           Searches within 'run_*/' subdirs first, then for loose '.pth' files."""
        best_model_found = None

        if not search_paths:
            logger.warning("No search paths provided to get_latest_model_path.")
            return None

        logger.info(f"Searching for models in: {search_paths}")

        # --- Search within run_* subdirectories first ---
        all_run_dirs = []
        for base_path in search_paths:
            if not os.path.isdir(base_path):
                logger.warning(f"Search path is not a directory: {base_path}")
                continue
            try:
                potential_runs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith('run_')]
                all_run_dirs.extend([os.path.join(base_path, d) for d in potential_runs])
            except OSError as e:
                logger.error(f"OSError accessing search path {base_path}: {e}")

        if all_run_dirs:
            logger.debug(f"Found {len(all_run_dirs)} potential run directories.")
            try:
                # Sort runs by modification time to check the latest first
                all_run_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                latest_run_dir = all_run_dirs[0]
                logger.info(f"Latest run directory found: {latest_run_dir}")

                # Find best or final model within the latest run
                model_files = [f for f in os.listdir(latest_run_dir) if (f.startswith('best_model') or f.startswith('final_model')) and f.endswith('.pth')]
                if model_files:
                    full_model_paths = [os.path.join(latest_run_dir, f) for f in model_files]
                    full_model_paths.sort(key=os.path.getmtime, reverse=True) # Get most recent model file
                    best_model_found = full_model_paths[0]
                    logger.info(f"Found latest model in run directory: {best_model_found}")
                else:
                    logger.warning(f"No best_model or final_model found in latest run dir: {latest_run_dir}")

            except Exception as e:
                logger.error(f"Error processing run directories: {e}", exc_info=True)
        else:
            logger.info("No 'run_*' subdirectories found in any search path.")

        # --- If no model found in run_* dirs, search for loose .pth files (e.g., default model) ---
        if best_model_found is None:
            logger.info("No model found in run_* dirs. Searching for loose .pth files in search paths...")
            all_loose_models = []
            for base_path in search_paths:
                if not os.path.isdir(base_path): continue # Skip if not dir
                try:
                    loose_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f)) and f.endswith('.pth')]
                    all_loose_models.extend(loose_files)
                except OSError as e:
                    logger.error(f"OSError searching for loose models in {base_path}: {e}")

            if all_loose_models:
                logger.info(f"Found {len(all_loose_models)} loose .pth files. Selecting latest.")
                all_loose_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                # Select the overall latest loose model found
                best_model_found = all_loose_models[0]
            else:
                logger.error("No run_* directories AND no loose .pth models found in any search path.")
                return None # Truly no model found

        logger.info(f"Selected final model file: {best_model_found}")
        return best_model_found

    def predict(self, image_path: str) -> Optional[Tuple[str, float]]:
        """Predict if an image file is real or fake using the optimized threshold."""
        try:
            logger.debug(f"Predicting image file: {image_path}")
            image = Image.open(image_path).convert('RGB')
            return self._predict_common(image) # Use common internal method
        except FileNotFoundError:
            logger.error(f"Prediction error: Image file not found at {image_path}")
            return None
        except Exception as e:
            logger.error(f"Error during prediction for file '{image_path}': {e}", exc_info=True)
            return None # Return None or a specific error tuple

    def predict_pil(self, image: Image.Image, cancel_event: threading.Event = None) -> Optional[Tuple[str, float]]:
        if cancel_event and cancel_event.is_set():
            logger.debug("Prediction cancelled before processing PIL image.")
            return None # Or a specific "cancelled" indicator
        try:
            logger.debug("Predicting PIL image")
            image_rgb = image.convert('RGB')
            # Pass cancel_event to _predict_common
            return self._predict_common(image_rgb, cancel_event)
        except Exception as e:
            logger.error(f"Error during prediction for PIL image: {e}", exc_info=True)
            return None

    def _predict_common(self, image: Image.Image, cancel_event: threading.Event = None) -> Optional[Tuple[str, float]]:
        if cancel_event and cancel_event.is_set():
            logger.debug("Prediction cancelled before tensor transformation.")
            return None

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)
            if cancel_event and cancel_event.is_set(): # Check after model inference
                logger.debug("Prediction cancelled after model inference.")
                return None
            probabilities = torch.softmax(output, dim=1)[0]
            probability_fake = probabilities[1].item()

        result = "FAKE" if probability_fake >= self.optimal_threshold else "REAL"
        logger.debug(f"Prediction: {result} (Fake Prob: {probability_fake:.4f}, Thresh: {self.optimal_threshold:.4f})")
        return result, probability_fake