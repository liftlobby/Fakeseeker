from facenet_pytorch import MTCNN
from PIL import Image, UnidentifiedImageError
import cv2
import numpy as np
import os
from typing import List, Tuple
import torch
import logging
import shutil
import threading
import subprocess
import json
import sys

logger = logging.getLogger(__name__)

class FaceExtractor:
    def __init__(self, device='cuda', min_confidence=0.9, min_face_size=50, config=None):
        """Initialize the MTCNN face detector with optimal parameters for deepfake detection."""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        if config:
            self.min_confidence = config.get('face_min_confidence', min_confidence)
            self.min_face_size = config.get('face_min_size', min_face_size)
        else:
            self.min_confidence = min_confidence
            self.min_face_size = min_face_size
            
        logger.info(f"Initializing MTCNN face detector on device: {self.device} "
                    f"with min_confidence={self.min_confidence}, min_face_size={self.min_face_size}")
        try:
            self.mtcnn = MTCNN(
                margin=14,
                keep_all=True,
                factor=0.7,
                device=self.device,
                thresholds=[0.6, 0.7, 0.7]
            )
        except Exception as e:
            logger.error(f"Failed to initialize MTCNN: {e}", exc_info=True)
            raise
        
        # --- Initialize ffprobe_path ---
        self.ffprobe_path = self._find_ffprobe()
    
    def _extract_from_pil(self, image_pil: Image.Image, cancel_event: threading.Event = None) -> List[Image.Image]:
        """Helper function to extract faces from a PIL image, checking for cancellation."""
        if cancel_event and cancel_event.is_set():
            logger.debug("PIL extraction cancelled at start.")
            return []
        
        extracted_faces_pil = []
        try:
            # Detect faces
            boxes, probs = self.mtcnn.detect(image_pil)

            if boxes is not None and probs is not None:
                for i, (box, confidence) in enumerate(zip(boxes, probs)):
                    # Periodically check for cancellation, e.g., every 5 faces
                    if cancel_event and cancel_event.is_set() and i % 5 == 0 :
                        logger.debug("PIL extraction cancelled during loop.")
                        return [] # Stop processing

                    if confidence >= self.min_confidence:
                        box_int = [int(b) for b in box]
                        width = box_int[2] - box_int[0]
                        height = box_int[3] - box_int[1]

                        if width >= self.min_face_size and height >= self.min_face_size:
                            face = image_pil.crop((box_int[0], box_int[1], box_int[2], box_int[3]))
                            extracted_faces_pil.append(face)
        except Exception as e:
            logger.error(f"Error during MTCNN detection or PIL cropping: {e}", exc_info=True)
            return [] # Return empty list on error
        return extracted_faces_pil

    def _filter_and_crop_faces(self, image_pil, boxes, probs):
        extracted_faces = []
        if boxes is None or probs is None:
            return extracted_faces

        for box, confidence in zip(boxes, probs):
            # Use self attributes for checks
            if confidence < self.min_confidence: continue
            box_int = [int(b) for b in box]
            w, h = box_int[2] - box_int[0], box_int[3] - box_int[1]
            if w < self.min_face_size or h < self.min_face_size: continue

            face = image_pil.crop((box_int[0], box_int[1], box_int[2], box_int[3]))
            extracted_faces.append(face)
        return extracted_faces

    def extract_faces_and_boxes_realtime(self, image: Image.Image) -> Tuple[List[Image.Image], List[List[int]]]:
        # This method is for real-time, cancellation is handled by stopping the feed thread
        # So, no cancel_event needed here unless you specifically want to make it interruptible
        # mid-detection on a single frame (usually not necessary).
        extracted_faces = []
        extracted_boxes = []
        try:
            image_rgb = image.convert('RGB')
            boxes, probs = self.mtcnn.detect(image_rgb)

            if boxes is not None and probs is not None:
                for box, confidence in zip(boxes, probs):
                    if confidence < self.min_confidence: continue
                    box_int = [int(b) for b in box]
                    x1, y1, x2, y2 = box_int
                    w, h = x2 - x1, y2 - y1
                    if w < self.min_face_size or h < self.min_face_size: continue
                    face = image_rgb.crop((x1, y1, x2, y2))
                    extracted_faces.append(face)
                    extracted_boxes.append(box_int)
            return extracted_faces, extracted_boxes
        except Exception as e:
            logger.error(f"Error extracting faces/boxes in real-time: {e}", exc_info=True)
            return [], []

    def extract_faces_from_image(self, image_path: str, cancel_event: threading.Event = None) -> List[Image.Image]:
        if cancel_event and cancel_event.is_set():
            logger.debug(f"Image extraction cancelled before opening: {image_path}")
            return []
        try:
            img = Image.open(image_path).convert('RGB')
            return self._extract_from_pil(img, cancel_event) # Pass cancel_event
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            return []
        except UnidentifiedImageError:
            logger.warning(f"Cannot identify image file (possibly corrupt): {image_path}")
            return []
        except Exception as e:
            logger.error(f"Error extracting faces from image '{image_path}': {e}", exc_info=True)
            return []

    
    def _find_ffprobe(self) -> str | None: # Using Python 3.10+ type hint
        """Tries to find ffprobe in PATH or common locations."""
        # For bundled app, ensure ffprobe is in a known relative location
        # and use a resource_path() function if it's defined globally or passed.
        # For simplicity here, we'll assume resource_path is available if needed.
        # If this script is part of a larger app, resource_path might be in main.py.
        
        # Helper for bundled path if running this file standalone for testing
        def local_resource_path(relative_path):
            try: base_path = sys._MEIPASS
            except Exception: base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # Assuming this file is in 'ui' or similar, go up one for project root
            return os.path.join(base_path, relative_path)

        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            # Path relative to the bundle's root
            # This assumes you've placed ffprobe (and ffmpeg + DLLs) in a 'bin' subdir of your bundle.
            # Adjust 'bin' if your spec file places it elsewhere (e.g., directly in root).
            bundled_ffprobe = local_resource_path(os.path.join('bin', 'ffprobe.exe' if sys.platform == 'win32' else 'ffprobe'))
            if os.path.exists(bundled_ffprobe):
                logger.info(f"Using bundled ffprobe: {bundled_ffprobe}")
                return bundled_ffprobe
        
        ffprobe_cmd = 'ffprobe.exe' if sys.platform == 'win32' else 'ffprobe'
        found_path = shutil.which(ffprobe_cmd)
        if found_path:
            logger.info(f"Found ffprobe in PATH: {found_path}")
            return found_path
        
        logger.warning("ffprobe not found in PATH or typical bundled locations. Video rotation metadata might not be available.")
        return None

    def get_video_rotation(self, video_path: str) -> int: # This returns the cv2.ROTATE_* constant or 0
        if not self.ffprobe_path:
            logger.warning("ffprobe path not set, cannot determine video rotation. Assuming 0.")
            return 0 
        try:
            cmd = [
                self.ffprobe_path, '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream_side_data=rotation', 
                '-show_entries', 'stream_tags=rotate', '-of', 'json', video_path
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, 
                                       creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
            stdout, stderr = process.communicate(timeout=15)

            if process.returncode != 0:
                logger.warning(f"ffprobe error for {os.path.basename(video_path)} (code {process.returncode}): {stderr.strip()}")
                return 0

            data = json.loads(stdout)
            rotation_cv2_const = 0 # Default to 0 (no rotation needed by cv2)
            processed_rotation_tag = False

            if data and 'streams' in data and len(data['streams']) > 0:
                stream = data['streams'][0]
                # Check side_data first (more common for display matrix rotation)
                if 'side_data_list' in stream:
                    for side_data in stream['side_data_list']:
                        if side_data.get('side_data_type') == 'Display Matrix': # Case sensitive
                            displaymatrix_rotation_str = side_data.get('rotation')
                            if displaymatrix_rotation_str is not None:
                                try:
                                    angle = int(float(displaymatrix_rotation_str))
                                    logger.info(f"Video {os.path.basename(video_path)}: ffprobe DisplayMatrix raw: {angle}")
                                    # If DisplayMatrix says -90 (player rotates CW), we need to rotate frames CW
                                    if angle == -90: rotation_cv2_const = cv2.ROTATE_90_CLOCKWISE
                                    # If DisplayMatrix says 90 (player rotates CCW), we need to rotate frames CCW
                                    elif angle == 90: rotation_cv2_const = cv2.ROTATE_90_COUNTERCLOCKWISE
                                    elif angle == 180 or angle == -180: rotation_cv2_const = cv2.ROTATE_180
                                    # ffprobe also uses 270 for -90 and -270 for 90 from some sources
                                    elif angle == 270: rotation_cv2_const = cv2.ROTATE_90_CLOCKWISE # Player rotates CW
                                    elif angle == -270: rotation_cv2_const = cv2.ROTATE_90_COUNTERCLOCKWISE # Player rotates CCW
                                    else: rotation_cv2_const = 0 
                                    processed_rotation_tag = True
                                    logger.info(f"Interpreted DisplayMatrix: cv2 rotation code {rotation_cv2_const if rotation_cv2_const !=0 else '0 (None)'}")
                                    return rotation_cv2_const # Return immediately if found
                                except ValueError:
                                    logger.warning(f"Could not parse DisplayMatrix rotation: {displaymatrix_rotation_str}")
                
                # Fallback to stream tags 'rotate' if no DisplayMatrix rotation processed
                if not processed_rotation_tag and 'tags' in stream and 'rotate' in stream['tags']:
                    try:
                        angle = int(float(stream['tags']['rotate']))
                        logger.info(f"Video {os.path.basename(video_path)}: ffprobe stream_tags raw: {angle}")
                        # 'tags.rotate' usually indicates how the frame is stored relative to "up".
                        # If it's 90, it's stored landscape but meant to be portrait.
                        # To make it upright for processing, we apply the specified rotation.
                        if angle == 90: rotation_cv2_const = cv2.ROTATE_90_CLOCKWISE
                        elif angle == 180: rotation_cv2_const = cv2.ROTATE_180
                        elif angle == 270: rotation_cv2_const = cv2.ROTATE_90_COUNTERCLOCKWISE
                        else: rotation_cv2_const = 0
                        processed_rotation_tag = True # Mark as processed here too
                        logger.info(f"Interpreted stream_tags: cv2 rotation code {rotation_cv2_const if rotation_cv2_const !=0 else '0 (None)'}")
                        return rotation_cv2_const # Return immediately
                    except ValueError:
                        logger.warning(f"Could not parse stream_tags rotation: {stream['tags']['rotate']}")
            
            if not processed_rotation_tag: # Log only if neither tag was found or parsed
                logger.info(f"No definitive rotation metadata found or parsed by ffprobe for {os.path.basename(video_path)}. Assuming 0.")
            return 0 # Default no rotation if nothing useful found
        except FileNotFoundError:
            logger.error(f"ffprobe command not found at '{self.ffprobe_path}'. Ensure ffmpeg is installed and in PATH, or correctly bundled.")
            self.ffprobe_path = None 
            return 0
        except subprocess.TimeoutExpired:
            logger.error(f"ffprobe timed out for {video_path}.")
            return 0
        except json.JSONDecodeError:
            logger.error(f"ffprobe output for {video_path} was not valid JSON. Output: {stdout}")
            return 0
        except Exception as e:
            logger.error(f"Error getting video rotation for {video_path} with ffprobe: {e}", exc_info=True)
            return 0

    def extract_faces_from_video(self, video_path: str, n_frames: int = 20,
                                 cancel_event: threading.Event = None,
                                 max_faces_to_return: int = None) -> List[Image.Image]:
        if cancel_event and cancel_event.is_set(): logger.debug(f"Vid extract cancelled: {video_path}"); return []
        extracted_faces, cap = [], None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): 
                logger.error(f"Vid open err: {video_path}")
                return []

            rotation_code = self.get_video_rotation(video_path) # Returns cv2.ROTATE_* constant or 0

            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_f <= 0: 
                logger.warning(f"Vid 0 frames: {video_path}")
                return []
            if n_frames <= 0: n_frames = min(max(1,total_f),20)
            actual_n = min(n_frames, total_f)
            if actual_n == 0: return []
            logger.info(f"Processing video: {video_path} ({total_f}f), sampling {actual_n}f, rotation_code to apply: {rotation_code if rotation_code !=0 else '0 (None)'}")
            frame_indices = np.linspace(0,total_f-1,actual_n,dtype=int)

            for i, frame_idx in enumerate(frame_indices):
                if cancel_event and cancel_event.is_set(): logger.info("Vid extract cancelled in loop."); break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret: 
                    logger.warning(f"Could not read frame {frame_idx} from {video_path}")
                    continue

                corrected_frame = frame # Initialize
                if rotation_code != 0:
                    logger.info(f"Applying rotation code {rotation_code} to frame {frame_idx}")
                    corrected_frame = cv2.rotate(frame, rotation_code)
                
                frame_rgb = cv2.cvtColor(corrected_frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                faces_from_frame = self._extract_from_pil(frame_pil, cancel_event)
                
                if max_faces_to_return is not None:
                    remaining_needed = max_faces_to_return - len(extracted_faces)
                    if remaining_needed <=0: break 
                    extracted_faces.extend(faces_from_frame[:remaining_needed])
                    if len(extracted_faces) >= max_faces_to_return:
                        logger.info(f"Reached max_faces_to_return ({max_faces_to_return}) from video.")
                        break 
                else:
                    extracted_faces.extend(faces_from_frame)

                if cancel_event and cancel_event.is_set(): logger.info("Vid extract cancelled post-frame-proc."); break
            
            if max_faces_to_return is not None and len(extracted_faces) > max_faces_to_return:
                extracted_faces = extracted_faces[:max_faces_to_return]
            return extracted_faces
        except Exception as e: logger.error(f"Vid extract err '{video_path}': {e}", exc_info=True); return []
        finally:
            if cap and cap.isOpened(): cap.release()

    def get_face_location(self, image) -> Tuple[int, int, int, int] | None: # Python 3.10+ type hint
        """ Get location of the *first* detected face. """
        try:
            image_rgb = image.convert('RGB')
            boxes, _ = self.mtcnn.detect(image_rgb) 
            if boxes is None or len(boxes) == 0:
                logger.debug("No faces found for get_face_location.")
                return None
            box = [int(b) for b in boxes[0]]
            return (box[1], box[2], box[3], box[0]) # top, right, bottom, left
        except Exception as e:
            logger.error(f"Error in get_face_location: {e}", exc_info=True)
            return None

    def cleanup(self):
        """Removes the temporary directory if it was used."""
        # If you removed self.temp_dir, this method might not be needed
        # or would do nothing. For now, assuming it might exist.
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temp directory '{self.temp_dir}': {e}", exc_info=True)