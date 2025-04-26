from facenet_pytorch import MTCNN
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
from typing import List, Optional, Tuple
import torch
import logging
import shutil

logger = logging.getLogger(__name__)

class FaceExtractor:
    def __init__(self, min_confidence=0.9, min_face_size=50):
        """Initialize the MTCNN face detector with optimal parameters for deepfake detection."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        logger.info(f"Initializing MTCNN face detector on device: {self.device}")
        try:
            self.mtcnn = MTCNN(
                margin=14,        # Add margin around faces
                keep_all=True,    # Keep all faces found
                factor=0.7,       # For faster detection
                device=self.device
            )
        except Exception as e:
            logger.error(f"Failed to initialize MTCNN: {e}", exc_info=True)
            raise

        # Create temporary directory for extracted faces
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Temporary directory for faces created: {self.temp_dir}")

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
        """
        Extract faces (as PIL images) and their bounding boxes from a single
        PIL image in real-time.
        Returns:
            Tuple containing:
                - List of PIL Image objects (cropped faces).
                - List of bounding boxes ([x1, y1, x2, y2]) relative to the input image.
        """
        extracted_faces = []
        extracted_boxes = [] # List to store corresponding boxes

        try:
            image_rgb = image.convert('RGB')
            # Keep probs if needed later, otherwise omit the second return value
            boxes, probs = self.mtcnn.detect(image_rgb)

            if boxes is not None and probs is not None:
                for box, confidence in zip(boxes, probs):
                    if confidence < self.min_confidence: continue

                    # Ensure box coords are ints for cropping/drawing
                    box_int = [int(b) for b in box]
                    x1, y1, x2, y2 = box_int # Unpack for clarity

                    w, h = x2 - x1, y2 - y1
                    if w < self.min_face_size or h < self.min_face_size:
                        continue

                    # Crop the face from the RGB image
                    face = image_rgb.crop((x1, y1, x2, y2))
                    extracted_faces.append(face)
                    # Add the integer box coordinates to the list
                    extracted_boxes.append(box_int)

            # Return both lists
            return extracted_faces, extracted_boxes

        except Exception as e:
            logger.error(f"Error extracting faces and boxes in real-time: {e}", exc_info=True)
            return [], [] # Return empty lists on error

    def extract_faces_from_image(self, image_path: str) -> List[Image.Image]:
        """
        Extract faces from a single image file.
        Returns list of PIL Image objects containing faces.
        """
        
        try:
            img = Image.open(image_path).convert('RGB')
            boxes, probs = self.mtcnn.detect(img)
            faces = self._filter_and_crop_faces(img, boxes, probs) # Get faces from helper

            if boxes is None or probs is None:
                logger.debug(f"No faces detected in image: {image_path}")
                return []

            logger.info(f"Extracted {len(extracted_faces)} faces from image: {image_path}")
            return faces

        except Exception as e:
            logger.error(f"Error extracting faces from image '{image_path}': {e}", exc_info=True)
            return []

    def extract_faces_from_video(self, video_path: str, n_frames: int = 20) -> List[Image.Image]:
        """
        Extract faces from video frames.
        Args:
            video_path: Path to video file
            n_frames: Number of frames to sample from video
        Returns list of PIL Image objects containing faces.
        """

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                return []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Processing video: {video_path} ({total_frames} frames)")

            if total_frames <= 0:
                 logger.warning(f"Video has 0 frames or failed to read count: {video_path}")
                 cap.release()
                 return []

            if n_frames > total_frames or n_frames <= 0:
                n_frames = total_frames
            frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
            logger.debug(f"Sampling {len(frame_indices)} frames from video.")

            extracted_faces = []
            processed_frames = 0
            all_video_faces = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Could not read frame {frame_idx} from video {video_path}")
                    continue
                processed_frames += 1

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)

                boxes, probs = self.mtcnn.detect(frame_pil)
                frame_faces = self._filter_and_crop_faces(frame_pil, boxes, probs)
                all_video_faces.extend(frame_faces)
                if boxes is None or probs is None:
                    continue

            cap.release()
            logger.info(f"Extracted {len(extracted_faces)} faces from {processed_frames} sampled frames in video: {video_path}")
            return all_video_faces
        except Exception as e:
            logger.error(f"Error extracting faces from video '{video_path}': {e}", exc_info=True)
            # Ensure cap is released even on error
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            return []

    def get_face_location(self, image) -> Optional[Tuple[int, int, int, int]]:
        """ Get location of the *first* detected face. """
        try:
            # Need to convert PIL Image to RGB for detect if not already
            image_rgb = image.convert('RGB')
            boxes, _ = self.mtcnn.detect(image_rgb) # Get boxes and optional probs
            if boxes is None or len(boxes) == 0:
                logger.debug("No faces found for get_face_location.")
                return None

            box = [int(b) for b in boxes[0]]  # Get first face
            logger.debug(f"First face location found: {box}")
            # Return (top, right, bottom, left)
            return (box[1], box[2], box[3], box[0])
        except Exception as e:
            logger.error(f"Error in get_face_location: {e}", exc_info=True) # Use logger
            return None

    def cleanup(self):
        """Removes the temporary directory."""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temp directory '{self.temp_dir}': {e}", exc_info=True)