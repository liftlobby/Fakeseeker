import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import sys
import os
import json
import logging
from typing import Tuple, Optional

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

        # ... (device setup) ...
        self.user_model_dir = user_model_dir
        self.user_threshold_path = user_threshold_path
        self.optimal_threshold = 0.5 # Default

        # --- Threshold Loading ---
        bundled_threshold_path = resource_path("optimal_threshold.json")
        threshold_to_load = None

        # Prioritize user path
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

        # --- Model Loading ---
        search_paths = []
        if self.user_model_dir and os.path.exists(self.user_model_dir):
             search_paths.append(self.user_model_dir)
             logger.info(f"Adding user model dir to search path: {self.user_model_dir}")
        # Example: Add fallback default model dir IF you bundle one
        # default_model_bundled_path = resource_path('default_model')
        # if os.path.exists(default_model_bundled_path):
        #      search_paths.append(default_model_bundled_path)
        #      logger.info(f"Adding default model dir to search path: {default_model_bundled_path}")

        model_path = self.get_latest_model_path(search_paths)

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
        """Finds the latest run dir and best model within provided search paths."""
        latest_run_dir = None
        latest_run_time = 0

        if not search_paths:
            logger.warning("No search paths provided to get_latest_model_path.")
            return None

        logger.info(f"Searching for models in: {search_paths}")
        all_run_dirs = []
        for base_path in search_paths:
            if not os.path.isdir(base_path):
                    logger.warning(f"Search path is not a directory: {base_path}")
                    continue
            try:
                    # Look for directories starting with 'run_'
                    potential_runs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith('run_')]
                    all_run_dirs.extend([os.path.join(base_path, d) for d in potential_runs])
            except OSError as e:
                    logger.error(f"OSError accessing search path {base_path}: {e}")

        if not all_run_dirs:
                logger.warning(f"No 'run_*' directories found in search paths.")
                # If no runs found, maybe check directly in user_model_dir for a loose .pth?
                # Or check default_model_dir for a specific default filename? (Added complexity)
                return None

        # Find the directory modified most recently (heuristic for latest run)
        try:
            all_run_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_run_dir = all_run_dirs[0]
            logger.info(f"Latest run directory found: {latest_run_dir}")
        except Exception as e:
                logger.error(f"Error sorting run directories by mtime: {e}")
                return None # Cannot determine latest

        # Find 'best_model*.pth' within the latest run directory
        try:
            model_files = [f for f in os.listdir(latest_run_dir) if f.startswith('best_model') and f.endswith('.pth')]
            if not model_files:
                logger.warning(f"No 'best_model*.pth' found in {latest_run_dir}.")
                # Fallback: Check for 'final_model*.pth'?
                model_files = [f for f in os.listdir(latest_run_dir) if f.startswith('final_model') and f.endswith('.pth')]
                if not model_files:
                        logger.error(f"No best_model or final_model found in {latest_run_dir}.")
                        return None

            # Sort model files (e.g., by modification time again) to get the absolute latest/best
            full_model_paths = [os.path.join(latest_run_dir, f) for f in model_files]
            full_model_paths.sort(key=os.path.getmtime, reverse=True)
            best_model_path = full_model_paths[0]
            logger.info(f"Selected model file: {best_model_path}")
            return best_model_path

        except Exception as e:
                logger.error(f"Error finding model file within {latest_run_dir}: {e}", exc_info=True)
                return None

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

    def predict_pil(self, image: Image.Image) -> Optional[Tuple[str, float]]:
        """Predict if a PIL image is real or fake using the optimized threshold."""
        try:
            logger.debug("Predicting PIL image")
            image_rgb = image.convert('RGB')
            return self._predict_common(image_rgb) # Use common internal method
        except Exception as e:
            logger.error(f"Error during prediction for PIL image: {e}", exc_info=True)
            return None # Return None or a specific error tuple

    def _predict_common(self, image: Image.Image) -> Tuple[str, float]:
        """Internal method to perform prediction on a PIL image."""
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            # Inside _predict_common
            probabilities = torch.softmax(output, dim=1)[0]
            # Assuming class 1 = FAKE, class 0 = REAL (verify based on training labels)
            probability_fake = probabilities[1].item()

        # Apply the threshold to the probability of the 'FAKE' class
        result = "FAKE" if probability_fake >= self.optimal_threshold else "REAL"
        logger.debug(f"Prediction: {result} (Fake Probability: {probability_fake:.4f}, Threshold: {self.optimal_threshold:.4f})")
        return result, probability_fake # Return label and raw probability of FAKE