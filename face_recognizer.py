"""
Face Recognizer Module
Uses face_recognition (dlib) for identity verification during exams
Handles face enrollment and real-time verification
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple, List
import config


class FaceRecognizer:
    """
    Face recognition for exam identity verification.
    
    Features:
    - Face enrollment (capture and store authorized face embedding)
    - Real-time verification (compare current face with authorized)
    - Status tracking (verified, unauthorized, no face)
    """
    
    # Verification status constants
    STATUS_VERIFIED = 'verified'
    STATUS_UNAUTHORIZED = 'unauthorized'
    STATUS_NO_FACE = 'no_face'
    STATUS_UNKNOWN = 'unknown'
    
    def __init__(self):
        # Always initialize authorized_encoding to avoid attribute errors
        self.authorized_encoding: Optional[np.ndarray] = None
        # Import face_recognition here to handle import errors gracefully
        try:
            import face_recognition
            self.face_recognition = face_recognition
            self.initialized = True
        except Exception as e:
            print("ERROR: face_recognition library not initialized!")
            import traceback
            traceback.print_exc()
            print(f"Actual import error: {e}")
            print("Install with: pip install face_recognition")
            self.initialized = False
            return
        
        # Storage path for authorized face
        self.faces_dir = os.path.join(os.path.dirname(__file__), 'faces')
        self.authorized_path = os.path.join(self.faces_dir, 'authorized.npy')
        
        # Ensure faces directory exists
        os.makedirs(self.faces_dir, exist_ok=True)
        
        # Authorized face embedding (loaded from file or set during registration)
        self.authorized_encoding: Optional[np.ndarray] = None
        
        # Current verification status
        self.current_status = self.STATUS_UNKNOWN
        
        # Frame counter for periodic verification
        self.frame_counter = 0
        
        # Load existing authorized face if available
        self._load_authorized_face()
    
    def _load_authorized_face(self) -> bool:
        """Load authorized face encoding from disk"""
        if os.path.exists(self.authorized_path):
            try:
                self.authorized_encoding = np.load(self.authorized_path)
                print(f"Loaded authorized face from {self.authorized_path}")
                return True
            except Exception as e:
                print(f"Error loading authorized face: {e}")
                self.authorized_encoding = None
        return False
    
    def is_face_registered(self) -> bool:
        """Check if an authorized face is registered"""
        return self.authorized_encoding is not None
    
    def delete_registration(self) -> bool:
        """Delete the registered face"""
        try:
            if os.path.exists(self.authorized_path):
                os.remove(self.authorized_path)
            self.authorized_encoding = None
            print("Face registration deleted")
            return True
        except Exception as e:
            print(f"Error deleting registration: {e}")
            return False
    
    def capture_face_encoding(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Capture face encoding from a frame.
        
        Args:
            frame: BGR image from webcam
            
        Returns:
            Tuple of (success, encoding, message)
        """
        if not self.initialized:
            return False, None, "Face recognition not initialized"
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = self.face_recognition.face_locations(rgb_frame)
        
        if len(face_locations) == 0:
            return False, None, "No face detected"
        
        if len(face_locations) > 1:
            return False, None, "Multiple faces detected - only one person allowed"
        
        # Get face encoding
        encodings = self.face_recognition.face_encodings(rgb_frame, face_locations)
        
        if len(encodings) == 0:
            return False, None, "Could not encode face"
        
        return True, encodings[0], "Face captured successfully"
    
    def register_face(self, frames: List[np.ndarray]) -> Tuple[bool, str]:
        """
        Register authorized face from multiple frames.
        Creates averaged embedding for robustness.
        
        Args:
            frames: List of BGR images containing the authorized face
            
        Returns:
            Tuple of (success, message)
        """
        if not self.initialized:
            return False, "Face recognition not initialized"
        
        encodings = []
        
        for i, frame in enumerate(frames):
            success, encoding, msg = self.capture_face_encoding(frame)
            if success and encoding is not None:
                encodings.append(encoding)
        
        if len(encodings) < config.FACE_MIN_SAMPLES:
            return False, f"Only captured {len(encodings)} valid faces. Need at least {config.FACE_MIN_SAMPLES}."
        
        # Average the encodings for robustness
        self.authorized_encoding = np.mean(encodings, axis=0)
        
        # Save to disk
        try:
            np.save(self.authorized_path, self.authorized_encoding)
            print(f"Saved authorized face to {self.authorized_path}")
            return True, f"Face registered successfully ({len(encodings)} samples)"
        except Exception as e:
            return False, f"Error saving face: {e}"
    
    def verify_face(self, frame: np.ndarray, force: bool = False) -> Tuple[str, float]:
        """
        Verify if current face matches authorized face.
        Only runs every FACE_VERIFY_INTERVAL frames unless forced.
        
        Args:
            frame: BGR image from webcam
            force: If True, run verification regardless of frame counter
            
        Returns:
            Tuple of (status, confidence/distance)
        """
        if not self.initialized:
            return self.STATUS_UNKNOWN, 0.0
        
        if self.authorized_encoding is None:
            return self.STATUS_UNKNOWN, 0.0
        
        # Check if we should run verification this frame
        self.frame_counter += 1
        if not force and self.frame_counter % config.FACE_VERIFY_INTERVAL != 0:
            return self.current_status, 0.0
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = self.face_recognition.face_locations(rgb_frame)
        
        if len(face_locations) == 0:
            self.current_status = self.STATUS_NO_FACE
            return self.STATUS_NO_FACE, 0.0
        
        # Get encodings for detected faces
        encodings = self.face_recognition.face_encodings(rgb_frame, face_locations)
        
        if len(encodings) == 0:
            self.current_status = self.STATUS_NO_FACE
            return self.STATUS_NO_FACE, 0.0
        
        # Check each detected face against authorized
        for encoding in encodings:
            distance = self.face_recognition.face_distance([self.authorized_encoding], encoding)[0]
            
            if distance <= config.FACE_MATCH_THRESHOLD:
                self.current_status = self.STATUS_VERIFIED
                return self.STATUS_VERIFIED, distance
        
        # No face matched - unauthorized person
        self.current_status = self.STATUS_UNAUTHORIZED
        # Return the distance of the closest face
        min_distance = min(
            self.face_recognition.face_distance([self.authorized_encoding], enc)[0]
            for enc in encodings
        )
        return self.STATUS_UNAUTHORIZED, min_distance
    
    def get_status(self) -> str:
        """Get current verification status"""
        return self.current_status
    
    def get_status_display(self) -> Tuple[str, Tuple[int, int, int]]:
        """
        Get status text and color for UI display.
        
        Returns:
            Tuple of (status_text, bgr_color)
        """
        if self.current_status == self.STATUS_VERIFIED:
            return "✓ Verified", config.COLORS['green']
        elif self.current_status == self.STATUS_UNAUTHORIZED:
            return "⚠ Unauthorized", config.COLORS['red']
        elif self.current_status == self.STATUS_NO_FACE:
            return "○ No Face", config.COLORS['yellow']
        else:
            return "? Unknown", config.COLORS['light_gray']
    
    def reset(self):
        """Reset verification state (for new exam session)"""
        self.current_status = self.STATUS_UNKNOWN
        self.frame_counter = 0
    
    def draw_face_box(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw bounding box around detected face with verification status.
        
        Args:
            frame: BGR image to draw on
            
        Returns:
            Frame with face box drawn
        """
        if not self.initialized:
            return frame
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = self.face_recognition.face_locations(rgb_frame)
        
        # Determine box color based on status
        if self.current_status == self.STATUS_VERIFIED:
            color = config.COLORS['green']
        elif self.current_status == self.STATUS_UNAUTHORIZED:
            color = config.COLORS['red']
        else:
            color = config.COLORS['yellow']
        
        # Draw boxes
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        return frame
