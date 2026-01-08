"""
Pose Detector Module
Uses MediaPipe for real-time pose, face, and hand landmark detection
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, Dict, List
import config


class PoseDetector:
    """
    Detects body pose, face mesh, and hands using MediaPipe
    Extracts key landmarks for behavior analysis
    """
    
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize detectors
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=config.MIN_FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=config.HAND_VISIBILITY_THRESHOLD,
            min_tracking_confidence=0.5
        )
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=config.MIN_FACE_DETECTION_CONFIDENCE
        )
        
        # Key landmark indices for face mesh
        self.NOSE_TIP = 1
        self.LEFT_EYE_INNER = 133
        self.RIGHT_EYE_INNER = 362
        self.LEFT_EYE_OUTER = 33
        self.RIGHT_EYE_OUTER = 263
        self.FOREHEAD = 10
        self.CHIN = 152
        self.LEFT_CHEEK = 234
        self.RIGHT_CHEEK = 454
        
        # Eye landmarks for gaze detection
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a video frame and extract all landmarks
        
        Args:
            frame: BGR image from webcam
            
        Returns:
            Dictionary containing all detection results
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = {
            'face_mesh': None,
            'pose': None,
            'hands': None,
            'face_detection': None,
            'face_count': 0,
            'head_angles': None,
            'gaze_direction': None,
            'hands_visible': {'left': False, 'right': False}
        }
        
        # Process face mesh
        face_mesh_results = self.face_mesh.process(rgb_frame)
        if face_mesh_results.multi_face_landmarks:
            results['face_mesh'] = face_mesh_results.multi_face_landmarks[0]
            results['head_angles'] = self._calculate_head_angles(
                face_mesh_results.multi_face_landmarks[0], 
                frame.shape
            )
            results['gaze_direction'] = self._calculate_gaze_direction(
                face_mesh_results.multi_face_landmarks[0],
                frame.shape
            )
        
        # Process pose
        pose_results = self.pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            results['pose'] = pose_results.pose_landmarks
        
        # Process hands
        hands_results = self.hands.process(rgb_frame)
        if hands_results.multi_hand_landmarks:
            results['hands'] = hands_results.multi_hand_landmarks
            for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                if idx < len(hands_results.multi_handedness):
                    handedness = hands_results.multi_handedness[idx].classification[0].label
                    if handedness == 'Left':
                        results['hands_visible']['right'] = True  # Mirrored
                    else:
                        results['hands_visible']['left'] = True  # Mirrored
        
        # Count faces
        face_detection_results = self.face_detection.process(rgb_frame)
        results['face_detection'] = face_detection_results
        if face_detection_results.detections:
            results['face_count'] = len(face_detection_results.detections)
        
        return results
    
    def _calculate_head_angles(self, face_landmarks, frame_shape: Tuple) -> Dict:
        """
        Calculate head orientation angles (yaw, pitch, roll)
        
        Args:
            face_landmarks: MediaPipe face landmarks
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            Dictionary with yaw, pitch, and roll angles in degrees
        """
        h, w = frame_shape[:2]
        
        # Get key landmark points
        nose = face_landmarks.landmark[self.NOSE_TIP]
        left_eye = face_landmarks.landmark[self.LEFT_EYE_INNER]
        right_eye = face_landmarks.landmark[self.RIGHT_EYE_INNER]
        forehead = face_landmarks.landmark[self.FOREHEAD]
        chin = face_landmarks.landmark[self.CHIN]
        left_cheek = face_landmarks.landmark[self.LEFT_CHEEK]
        right_cheek = face_landmarks.landmark[self.RIGHT_CHEEK]
        
        # Convert to pixel coordinates
        nose_pt = np.array([nose.x * w, nose.y * h, nose.z * w])
        left_eye_pt = np.array([left_eye.x * w, left_eye.y * h, left_eye.z * w])
        right_eye_pt = np.array([right_eye.x * w, right_eye.y * h, right_eye.z * w])
        forehead_pt = np.array([forehead.x * w, forehead.y * h, forehead.z * w])
        chin_pt = np.array([chin.x * w, chin.y * h, chin.z * w])
        left_cheek_pt = np.array([left_cheek.x * w, left_cheek.y * h, left_cheek.z * w])
        right_cheek_pt = np.array([right_cheek.x * w, right_cheek.y * h, right_cheek.z * w])
        
        # Calculate yaw (horizontal head turn)
        eye_center = (left_eye_pt + right_eye_pt) / 2
        face_width = np.linalg.norm(left_cheek_pt[:2] - right_cheek_pt[:2])
        nose_offset = nose_pt[0] - eye_center[0]
        yaw = np.arcsin(np.clip(nose_offset / (face_width / 2 + 1e-6), -1, 1)) * 180 / np.pi
        
        # Calculate pitch (vertical head tilt)
        face_height = np.linalg.norm(forehead_pt[:2] - chin_pt[:2])
        vertical_offset = nose_pt[1] - eye_center[1]
        pitch = np.arcsin(np.clip(vertical_offset / (face_height / 2 + 1e-6), -1, 1)) * 180 / np.pi
        
        # Calculate roll (head tilt sideways)
        eye_vector = right_eye_pt[:2] - left_eye_pt[:2]
        roll = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi
        
        return {
            'yaw': yaw,      # Positive = looking right, Negative = looking left
            'pitch': pitch,  # Positive = looking down, Negative = looking up
            'roll': roll     # Positive = tilted right, Negative = tilted left
        }
    
    def _calculate_gaze_direction(self, face_landmarks, frame_shape: Tuple) -> Dict:
        """
        Calculate eye gaze direction using iris landmarks
        
        Args:
            face_landmarks: MediaPipe face landmarks
            frame_shape: Shape of the frame
            
        Returns:
            Dictionary with gaze deviation information
        """
        h, w = frame_shape[:2]
        
        try:
            # Get iris centers
            left_iris_pts = [face_landmarks.landmark[i] for i in self.LEFT_IRIS]
            right_iris_pts = [face_landmarks.landmark[i] for i in self.RIGHT_IRIS]
            
            left_iris_center = np.mean([[p.x, p.y] for p in left_iris_pts], axis=0)
            right_iris_center = np.mean([[p.x, p.y] for p in right_iris_pts], axis=0)
            
            # Get eye corners for reference
            left_eye_inner = face_landmarks.landmark[self.LEFT_EYE_INNER]
            left_eye_outer = face_landmarks.landmark[self.LEFT_EYE_OUTER]
            right_eye_inner = face_landmarks.landmark[self.RIGHT_EYE_INNER]
            right_eye_outer = face_landmarks.landmark[self.RIGHT_EYE_OUTER]
            
            # Calculate horizontal gaze deviation for each eye
            left_eye_width = abs(left_eye_outer.x - left_eye_inner.x)
            left_eye_center_x = (left_eye_outer.x + left_eye_inner.x) / 2
            left_gaze_offset = (left_iris_center[0] - left_eye_center_x) / (left_eye_width + 1e-6)
            
            right_eye_width = abs(right_eye_outer.x - right_eye_inner.x)
            right_eye_center_x = (right_eye_outer.x + right_eye_inner.x) / 2
            right_gaze_offset = (right_iris_center[0] - right_eye_center_x) / (right_eye_width + 1e-6)
            
            # Average gaze deviation
            avg_gaze_offset = (left_gaze_offset + right_gaze_offset) / 2
            gaze_angle = avg_gaze_offset * 45  # Convert to approximate degrees
            
            return {
                'horizontal_offset': avg_gaze_offset,
                'angle': gaze_angle,
                'looking_left': gaze_angle < -config.GAZE_DEVIATION_THRESHOLD,
                'looking_right': gaze_angle > config.GAZE_DEVIATION_THRESHOLD,
                'looking_center': abs(gaze_angle) <= config.GAZE_DEVIATION_THRESHOLD
            }
        except Exception:
            return {
                'horizontal_offset': 0,
                'angle': 0,
                'looking_left': False,
                'looking_right': False,
                'looking_center': True
            }
    
    def draw_landmarks(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw detected landmarks on the frame
        
        Args:
            frame: Original BGR frame
            results: Detection results from process_frame
            
        Returns:
            Frame with landmarks drawn
        """
        annotated_frame = frame.copy()
        
        if not config.SHOW_LANDMARKS:
            return annotated_frame
        
        # Draw face mesh
        if results['face_mesh']:
            self.mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=results['face_mesh'],
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # Draw iris landmarks
            self.mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=results['face_mesh'],
                connections=self.mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )
        
        # Draw pose landmarks
        if results['pose']:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results['pose'],
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw hand landmarks
        if results['hands']:
            for hand_landmarks in results['hands']:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return annotated_frame
    
    def release(self):
        """Release MediaPipe resources"""
        self.face_mesh.close()
        self.pose.close()
        self.hands.close()
        self.face_detection.close()
