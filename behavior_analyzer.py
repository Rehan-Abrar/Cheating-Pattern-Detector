"""
Behavior Analyzer Module
Analyzes pose detection results to identify suspicious behaviors
Uses rule-based detection with configurable thresholds
Returns detected events for cumulative risk scoring
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import config


@dataclass
class DetectedBehavior:
    """Represents a detected suspicious behavior"""
    behavior_type: str
    detected: bool
    details: Dict = None


class BehaviorAnalyzer:
    """
    Analyzes pose detection results to identify suspicious exam behaviors.
    Detects events and reports them for cumulative scoring.
    """
    
    def __init__(self):
        # Behavior state tracking (for duration-based detection)
        self.head_turn_start: Optional[float] = None
        self.no_face_start: Optional[float] = None
        self.gaze_deviation_start: Optional[float] = None
        self.multiple_faces_start: Optional[float] = None
        self.looking_away_start: Optional[float] = None
        
        # Current active states (for UI display)
        self.current_states: Dict[str, bool] = {
            'head_turn': False,
            'gaze_deviation': False,
            'multiple_faces': False,
            'no_face': False,
            'looking_away': False
        }
    
    def analyze(self, detection_results: Dict) -> Dict[str, DetectedBehavior]:
        """
        Analyze detection results and identify suspicious behaviors.
        
        Args:
            detection_results: Results from PoseDetector.process_frame()
            
        Returns:
            Dictionary of detected behaviors with their status
        """
        current_time = time.time()
        detected_behaviors = {}
        
        # Check each behavior type
        detected_behaviors['head_turn'] = self._check_head_turn(detection_results, current_time)
        detected_behaviors['gaze_deviation'] = self._check_gaze_deviation(detection_results, current_time)
        detected_behaviors['multiple_faces'] = self._check_multiple_faces(detection_results, current_time)
        detected_behaviors['no_face'] = self._check_no_face(detection_results, current_time)
        detected_behaviors['looking_away'] = self._check_looking_away(detection_results, current_time)
        
        return detected_behaviors
    
    def get_current_states(self) -> Dict[str, bool]:
        """Get current active behavior states for UI display"""
        return self.current_states.copy()
    
    def _check_head_turn(self, results: Dict, current_time: float) -> DetectedBehavior:
        """Check for suspicious head turns"""
        head_angles = results.get('head_angles')
        
        if head_angles is None:
            self.head_turn_start = None
            self.current_states['head_turn'] = False
            return DetectedBehavior('head_turn', False)
        
        yaw = abs(head_angles['yaw'])
        
        if yaw > config.HEAD_TURN_THRESHOLD:
            if self.head_turn_start is None:
                self.head_turn_start = current_time
            
            duration = current_time - self.head_turn_start
            self.current_states['head_turn'] = True
            
            if duration >= config.HEAD_TURN_DURATION:
                # Reset timer after detection to allow re-detection
                self.head_turn_start = current_time
                return DetectedBehavior(
                    'head_turn', 
                    True, 
                    {'yaw': head_angles['yaw'], 'direction': 'right' if head_angles['yaw'] > 0 else 'left'}
                )
        else:
            self.head_turn_start = None
            self.current_states['head_turn'] = False
        
        return DetectedBehavior('head_turn', False)
    
    def _check_gaze_deviation(self, results: Dict, current_time: float) -> DetectedBehavior:
        """Check for eye gaze deviation from screen"""
        gaze_direction = results.get('gaze_direction')
        
        if gaze_direction is None:
            self.gaze_deviation_start = None
            self.current_states['gaze_deviation'] = False
            return DetectedBehavior('gaze_deviation', False)
        
        if gaze_direction['looking_left'] or gaze_direction['looking_right']:
            if self.gaze_deviation_start is None:
                self.gaze_deviation_start = current_time
            
            duration = current_time - self.gaze_deviation_start
            self.current_states['gaze_deviation'] = True
            
            if duration >= config.HEAD_TURN_DURATION * 0.5:  # Shorter threshold for gaze
                self.gaze_deviation_start = current_time
                direction = 'left' if gaze_direction['looking_left'] else 'right'
                return DetectedBehavior(
                    'gaze_deviation', 
                    True, 
                    {'angle': gaze_direction['angle'], 'direction': direction}
                )
        else:
            self.gaze_deviation_start = None
            self.current_states['gaze_deviation'] = False
        
        return DetectedBehavior('gaze_deviation', False)
    
    def _check_multiple_faces(self, results: Dict, current_time: float) -> DetectedBehavior:
        """Check for multiple faces in the frame"""
        face_count = results.get('face_count', 0)
        
        if face_count > 1 and config.MULTIPLE_FACES_ALERT:
            if self.multiple_faces_start is None:
                self.multiple_faces_start = current_time
            
            duration = current_time - self.multiple_faces_start
            self.current_states['multiple_faces'] = True
            
            if duration >= 0.5:  # Brief delay to avoid false positives
                self.multiple_faces_start = current_time
                return DetectedBehavior(
                    'multiple_faces', 
                    True, 
                    {'face_count': face_count}
                )
        else:
            self.multiple_faces_start = None
            self.current_states['multiple_faces'] = False
        
        return DetectedBehavior('multiple_faces', False)
    
    def _check_no_face(self, results: Dict, current_time: float) -> DetectedBehavior:
        """Check if no face is detected in frame"""
        face_count = results.get('face_count', 0)
        face_mesh = results.get('face_mesh')
        
        if face_count == 0 or face_mesh is None:
            if self.no_face_start is None:
                self.no_face_start = current_time
            
            duration = current_time - self.no_face_start
            self.current_states['no_face'] = True
            
            if duration >= config.NO_FACE_DURATION:
                self.no_face_start = current_time
                return DetectedBehavior(
                    'no_face', 
                    True, 
                    {'duration': duration}
                )
        else:
            self.no_face_start = None
            self.current_states['no_face'] = False
        
        return DetectedBehavior('no_face', False)
        
        return DetectedBehavior('no_face', False)
    
    def _check_looking_away(self, results: Dict, current_time: float) -> DetectedBehavior:
        """Combined check for looking away from screen (head + gaze)"""
        head_angles = results.get('head_angles')
        gaze_direction = results.get('gaze_direction')
        
        head_turned = False
        gaze_off = False
        
        if head_angles:
            head_turned = abs(head_angles['yaw']) > config.HEAD_TURN_THRESHOLD * 0.7
        
        if gaze_direction:
            gaze_off = not gaze_direction['looking_center']
        
        if head_turned and gaze_off:
            if self.looking_away_start is None:
                self.looking_away_start = current_time
            
            duration = current_time - self.looking_away_start
            self.current_states['looking_away'] = True
            
            if duration >= config.HEAD_TURN_DURATION * 0.8:
                self.looking_away_start = current_time
                return DetectedBehavior(
                    'looking_away', 
                    True, 
                    {
                        'head_yaw': head_angles['yaw'] if head_angles else 0,
                        'gaze_angle': gaze_direction['angle'] if gaze_direction else 0
                    }
                )
        else:
            self.looking_away_start = None
            self.current_states['looking_away'] = False
        
        return DetectedBehavior('looking_away', False)
    
    def reset(self):
        """Reset all behavior tracking"""
        self.head_turn_start = None
        self.no_face_start = None
        self.gaze_deviation_start = None
        self.multiple_faces_start = None
        self.looking_away_start = None
        
        for key in self.current_states:
            self.current_states[key] = False
