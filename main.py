import cv2
import numpy as np
import time
import sys
from typing import Optional, Tuple
import config
from pose_detector import PoseDetector
from behavior_analyzer import BehaviorAnalyzer
from risk_scorer import RiskScorer


class Button:
    """Simple clickable button for OpenCV interface"""
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 text: str, color: Tuple, hover_color: Tuple, text_color: Tuple = (255, 255, 255)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False
    
    def contains(self, px: int, py: int) -> bool:
        """Check if point is inside button"""
        return (self.x <= px <= self.x + self.width and 
                self.y <= py <= self.y + self.height)
    
    def draw(self, frame: np.ndarray):
        """Draw button on frame"""
        color = self.hover_color if self.is_hovered else self.color
        
        # Draw button background with rounded appearance
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), 
                     color, -1)
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), 
                     (255, 255, 255), 2)
        
        # Draw text centered
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(self.text, font, font_scale, thickness)[0]
        text_x = self.x + (self.width - text_size[0]) // 2
        text_y = self.y + (self.height + text_size[1]) // 2
        cv2.putText(frame, self.text, (text_x, text_y), font, font_scale, 
                   self.text_color, thickness)


class ExamMonitor:
    """
    Main application class for the Cheating Pattern Detector.
    Manages the complete exam monitoring workflow:
    1. Start Screen (with Start button)
    2. Monitoring Screen (full-screen with Stop button)
    3. Results Screen (final summary with optional graph)
    """
    
    # Application states
    STATE_START = 'start'
    STATE_MONITORING = 'monitoring'
    STATE_RESULTS = 'results'
    
    def __init__(self):
        # Current application state
        self.state = self.STATE_START
        
        # Components (initialized when needed)
        self.pose_detector: Optional[PoseDetector] = None
        self.behavior_analyzer: Optional[BehaviorAnalyzer] = None
        self.risk_scorer: Optional[RiskScorer] = None
        
        # Video capture
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Window settings - will be updated based on actual screen
        self.window_name = 'AI Exam Proctoring System'
        self.screen_width = 1280
        self.screen_height = 720
        
        # Mouse position for button hover
        self.mouse_x = 0
        self.mouse_y = 0
        
        # Session results (stored after exam ends)
        self.session_results = None
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
    
    def run(self):
        """Main application loop"""
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Try to get screen resolution
        try:
            import ctypes
            user32 = ctypes.windll.user32
            self.screen_width = user32.GetSystemMetrics(0)
            self.screen_height = user32.GetSystemMetrics(1)
        except:
            self.screen_width = 1280
            self.screen_height = 720
        
        # Set window size and fullscreen
        cv2.resizeWindow(self.window_name, self.screen_width, self.screen_height)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Set mouse callback
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
        
        running = True
        while running:
            if self.state == self.STATE_START:
                frame = self._render_start_screen()
            elif self.state == self.STATE_MONITORING:
                frame = self._render_monitoring_screen()
            elif self.state == self.STATE_RESULTS:
                frame = self._render_results_screen()
            else:
                break
            
            cv2.imshow(self.window_name, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                if self.state == self.STATE_MONITORING:
                    self._stop_monitoring()
                elif self.state == self.STATE_RESULTS:
                    running = False
                else:
                    running = False
            elif key == ord('q'):
                if self.state != self.STATE_MONITORING:
                    running = False
            elif key == ord('r') and self.state == self.STATE_RESULTS:
                # Restart - go back to start screen
                self.state = self.STATE_START
                self.session_results = None
            elif key == ord('s') or key == ord(' '):  # S or SPACE to start
                if self.state == self.STATE_START:
                    self._start_monitoring()
                elif self.state == self.STATE_MONITORING:
                    self._stop_monitoring()
        
        self._cleanup()
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        self.mouse_x = x
        self.mouse_y = y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Click at ({x}, {y}) - State: {self.state}")
            self._handle_click(x, y)
    
    def _handle_click(self, x: int, y: int):
        """Handle mouse click based on current state"""
        if self.state == self.STATE_START:
            # Check if Start button was clicked
            btn = self._get_start_button()
            print(f"Start button: x={btn.x}, y={btn.y}, w={btn.width}, h={btn.height}")
            print(f"Contains click: {btn.contains(x, y)}")
            if btn.contains(x, y):
                self._start_monitoring()
        
        elif self.state == self.STATE_MONITORING:
            # Check if Stop button was clicked
            btn = self._get_stop_button()
            if btn.contains(x, y):
                self._stop_monitoring()
        
        elif self.state == self.STATE_RESULTS:
            # Check for New Exam or Exit buttons
            new_btn, exit_btn = self._get_results_buttons()
            if new_btn.contains(x, y):
                self.state = self.STATE_START
                self.session_results = None
            elif exit_btn.contains(x, y):
                self._cleanup()
                sys.exit(0)
    
    def _get_start_button(self) -> Button:
        """Get the Start Exam button"""
        btn_width, btn_height = 250, 60
        btn_x = (self.screen_width - btn_width) // 2
        btn_y = (self.screen_height - btn_height) // 2 + 50
        return Button(btn_x, btn_y, btn_width, btn_height, "START EXAM",
                     config.BUTTON_COLOR, config.BUTTON_HOVER_COLOR)
    
    def _get_stop_button(self) -> Button:
        """Get the Stop Exam button"""
        btn_width, btn_height = 150, 45
        btn_x = self.screen_width - btn_width - 20
        btn_y = 20
        return Button(btn_x, btn_y, btn_width, btn_height, "STOP EXAM",
                     config.STOP_BUTTON_COLOR, config.STOP_BUTTON_HOVER)
    
    def _get_results_buttons(self) -> Tuple[Button, Button]:
        """Get the New Exam and Exit buttons for results screen"""
        btn_width, btn_height = 180, 50
        spacing = 30
        total_width = btn_width * 2 + spacing
        start_x = (self.screen_width - total_width) // 2
        btn_y = self.screen_height - 100
        
        new_btn = Button(start_x, btn_y, btn_width, btn_height, "NEW EXAM",
                        config.BUTTON_COLOR, config.BUTTON_HOVER_COLOR)
        exit_btn = Button(start_x + btn_width + spacing, btn_y, btn_width, btn_height, "EXIT",
                         config.STOP_BUTTON_COLOR, config.STOP_BUTTON_HOVER)
        return new_btn, exit_btn
    
    def _start_monitoring(self):
        """Start the exam monitoring session"""
        print("\n" + "=" * 60)
        print("    STARTING EXAM MONITORING")
        print("=" * 60)
        
        # Initialize components
        print("Initializing detection components...")
        self.pose_detector = PoseDetector()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.risk_scorer = RiskScorer()
        
        # Start webcam
        print("Connecting to webcam...")
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        
        if not self.cap.isOpened():
            print("ERROR: Could not open webcam!")
            return
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        # Start risk scoring session
        self.risk_scorer.start_session()
        
        # Reset FPS counter
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        print("Monitoring started!")
        print("-" * 60)
        
        self.state = self.STATE_MONITORING
    
    def _stop_monitoring(self):
        """Stop the exam monitoring session"""
        print("\n" + "=" * 60)
        print("    STOPPING EXAM MONITORING")
        print("=" * 60)
        
        # End risk scoring session
        if self.risk_scorer:
            self.risk_scorer.end_session()
            self.session_results = self.risk_scorer.get_session_summary()
        
        # Release webcam
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Release pose detector
        if self.pose_detector:
            self.pose_detector.release()
            self.pose_detector = None
        
        print("Monitoring stopped.")
        print("-" * 60)
        
        self.state = self.STATE_RESULTS
    
    def _render_start_screen(self) -> np.ndarray:
        """Render the start screen with Start button"""
        frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        frame[:] = config.COLORS['dark_gray']
        
        # Title
        title = "AI EXAM PROCTORING SYSTEM"
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_size = cv2.getTextSize(title, font, 1.5, 3)[0]
        title_x = (self.screen_width - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, 150), font, 1.5, config.COLORS['white'], 3)
        
        # Subtitle
        subtitle = "Cheating Pattern Detector"
        sub_size = cv2.getTextSize(subtitle, font, 0.8, 2)[0]
        sub_x = (self.screen_width - sub_size[0]) // 2
        cv2.putText(frame, subtitle, (sub_x, 200), font, 0.8, config.COLORS['cyan'], 2)
        
        # Instructions
        instructions = [
            "This system will monitor your exam session using webcam.",
            "Suspicious behaviors will be detected and logged.",
            "",
            "Detected behaviors:",
            "  - Head turns beyond threshold",
            "  - Gaze deviation (looking away)",
            "  - Multiple faces detected",
            "  - No face in frame",
            "  - Hands not visible",
            "",
            "Click 'START EXAM' when ready to begin."
        ]
        
        y_offset = 280
        for line in instructions:
            if line:
                cv2.putText(frame, line, (200, y_offset), font, 0.6, 
                           config.COLORS['light_gray'], 1)
            y_offset += 30
        
        # Draw Start button
        btn = self._get_start_button()
        btn.is_hovered = btn.contains(self.mouse_x, self.mouse_y)
        btn.draw(frame)
        
        # Footer
        footer = "Press ESC to exit | Press SPACE or S to start"
        foot_size = cv2.getTextSize(footer, font, 0.5, 1)[0]
        foot_x = (self.screen_width - foot_size[0]) // 2
        cv2.putText(frame, footer, (foot_x, self.screen_height - 30), 
                   font, 0.5, config.COLORS['light_gray'], 1)
        
        return frame
    
    def _render_monitoring_screen(self) -> np.ndarray:
        """Render the monitoring screen with live webcam feed"""
        # Read frame from webcam
        if self.cap is None or not self.cap.isOpened():
            self._stop_monitoring()
            return np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        ret, frame = self.cap.read()
        if not ret:
            self._stop_monitoring()
            return np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Resize to screen size
        frame = cv2.resize(frame, (self.screen_width, self.screen_height))
        
        # Process frame through pose detector
        detection_results = self.pose_detector.process_frame(frame)
        
        # Analyze behaviors
        detected_behaviors = self.behavior_analyzer.analyze(detection_results)
        
        # Check for new events and add to risk scorer
        for behavior_type, behavior in detected_behaviors.items():
            if behavior.detected:
                self.risk_scorer.add_event(behavior_type)
        
        # Draw landmarks
        if config.SHOW_LANDMARKS:
            frame = self.pose_detector.draw_landmarks(frame, detection_results)
        
        # Draw overlay UI
        frame = self._draw_monitoring_overlay(frame, detection_results, detected_behaviors)
        
        # Update FPS
        self._update_fps()
        
        return frame
    
    def _draw_monitoring_overlay(self, frame: np.ndarray, detection_results: dict, 
                                  detected_behaviors: dict) -> np.ndarray:
        """Draw monitoring UI overlay on the frame"""
        h, w = frame.shape[:2]
        
        # Draw semi-transparent panel for risk score (top-left)
        self._draw_panel(frame, 10, 10, 280, 140)
        
        # Risk Score
        score = self.risk_scorer.get_current_score()
        risk_level = self.risk_scorer.get_risk_level()
        
        cv2.putText(frame, "RISK SCORE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, config.COLORS['white'], 2)
        
        # Score color based on level
        if risk_level == 'low':
            score_color = config.COLORS['green']
        elif risk_level == 'medium':
            score_color = config.COLORS['yellow']
        else:
            score_color = config.COLORS['red']
        
        cv2.putText(frame, f"{score}%", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.5, score_color, 3)
        
        cv2.putText(frame, f"Level: {risk_level.upper()}", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color, 2)
        
        # Progress bar
        bar_x, bar_y, bar_w, bar_h = 20, 130, 250, 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), 
                     config.COLORS['white'], 1)
        fill_w = int((score / 100) * bar_w)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), 
                     score_color, -1)
        
        # Draw behavior status panel (top-left, below risk score)
        self._draw_panel(frame, 10, 160, 280, 180)
        cv2.putText(frame, "BEHAVIOR STATUS", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, config.COLORS['white'], 2)
        
        current_states = self.behavior_analyzer.get_current_states()
        behaviors = [
            ('Head Turn', 'head_turn'),
            ('Gaze Deviation', 'gaze_deviation'),
            ('Multiple Faces', 'multiple_faces'),
            ('No Face', 'no_face'),
            ('Hands Hidden', 'hand_missing'),
            ('Looking Away', 'looking_away')
        ]
        
        y_offset = 210
        for label, key in behaviors:
            is_active = current_states.get(key, False)
            color = config.COLORS['red'] if is_active else config.COLORS['green']
            cv2.circle(frame, (30, y_offset - 5), 8, color, -1)
            cv2.putText(frame, label, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, config.COLORS['white'], 1)
            y_offset += 25
        
        # Draw event log panel (bottom-left)
        self._draw_panel(frame, 10, h - 220, 350, 210)
        cv2.putText(frame, "EVENT LOG", (20, h - 195), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, config.COLORS['white'], 2)
        
        recent_events = self.risk_scorer.get_recent_events()
        y_offset = h - 170
        for event_str in recent_events:
            cv2.putText(frame, event_str, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.45, config.COLORS['yellow'], 1)
            y_offset += 22
        
        # Draw elapsed time (top-right area)
        elapsed = self.risk_scorer.get_formatted_time()
        cv2.putText(frame, f"Time: {elapsed}", (w - 180, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, config.COLORS['white'], 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps}", (w - 180, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, config.COLORS['white'], 1)
        
        # Draw face count
        face_count = detection_results.get('face_count', 0)
        face_color = config.COLORS['green'] if face_count == 1 else config.COLORS['red']
        cv2.putText(frame, f"Faces: {face_count}", (w - 180, 140), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, face_color, 1)
        
        # Draw Stop button
        btn = self._get_stop_button()
        btn.is_hovered = btn.contains(self.mouse_x, self.mouse_y)
        btn.draw(frame)
        
        # Draw head angles if available
        if detection_results.get('head_angles'):
            self._draw_head_angles(frame, detection_results['head_angles'])
        
        return frame
    
    def _draw_panel(self, frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.7):
        """Draw a semi-transparent panel"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), config.COLORS['white'], 1)
    
    def _draw_head_angles(self, frame: np.ndarray, head_angles: dict):
        """Draw head orientation indicator"""
        h, w = frame.shape[:2]
        
        self._draw_panel(frame, w - 200, h - 100, 190, 90)
        
        cv2.putText(frame, "HEAD ORIENTATION", (w - 190, h - 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLORS['white'], 1)
        
        yaw = head_angles['yaw']
        pitch = head_angles['pitch']
        
        yaw_color = config.COLORS['red'] if abs(yaw) > config.HEAD_TURN_THRESHOLD else config.COLORS['green']
        
        cv2.putText(frame, f"Yaw: {yaw:+.1f}", (w - 190, h - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, yaw_color, 1)
        cv2.putText(frame, f"Pitch: {pitch:+.1f}", (w - 190, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLORS['white'], 1)
    
    def _render_results_screen(self) -> np.ndarray:
        """Render the final results screen"""
        frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        frame[:] = config.COLORS['dark_gray']
        
        if self.session_results is None:
            return frame
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Title
        title = "EXAM SESSION COMPLETE"
        title_size = cv2.getTextSize(title, font, 1.2, 3)[0]
        title_x = (self.screen_width - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, 60), font, 1.2, config.COLORS['white'], 3)
        
        # Results panel
        panel_x, panel_y = 100, 100
        panel_w, panel_h = 500, 400
        self._draw_panel(frame, panel_x, panel_y, panel_w, panel_h, 0.5)
        
        # Final Score
        score = self.session_results['final_score']
        risk_level = self.session_results['risk_level']
        
        if risk_level == 'low':
            score_color = config.COLORS['green']
        elif risk_level == 'medium':
            score_color = config.COLORS['yellow']
        else:
            score_color = config.COLORS['red']
        
        cv2.putText(frame, "FINAL RISK SCORE", (panel_x + 20, panel_y + 40), 
                   font, 0.8, config.COLORS['white'], 2)
        cv2.putText(frame, f"{score}%", (panel_x + 20, panel_y + 100), 
                   font, 2.0, score_color, 4)
        cv2.putText(frame, f"Risk Level: {risk_level.upper()}", (panel_x + 20, panel_y + 140), 
                   font, 0.8, score_color, 2)
        
        # Duration
        duration = self.session_results['duration_formatted']
        cv2.putText(frame, f"Exam Duration: {duration}", (panel_x + 20, panel_y + 190), 
                   font, 0.7, config.COLORS['white'], 2)
        
        # Total events
        total_events = self.session_results['total_events']
        cv2.putText(frame, f"Total Suspicious Events: {total_events}", (panel_x + 20, panel_y + 230), 
                   font, 0.7, config.COLORS['white'], 2)
        
        # Event breakdown
        cv2.putText(frame, "Event Breakdown:", (panel_x + 20, panel_y + 280), 
                   font, 0.6, config.COLORS['cyan'], 2)
        
        event_counts = self.session_results['event_counts']
        y_offset = panel_y + 310
        event_names = {
            'multiple_faces': 'Multiple Faces',
            'no_face': 'No Face',
            'head_turn': 'Head Turn',
            'gaze_deviation': 'Gaze Deviation',
            'hand_missing': 'Hands Hidden',
            'looking_away': 'Looking Away'
        }
        
        for key, name in event_names.items():
            count = event_counts.get(key, 0)
            weight = config.RISK_WEIGHTS.get(key, 0)
            if count > 0:
                cv2.putText(frame, f"  {name}: {count} (+{weight} each)", 
                           (panel_x + 30, y_offset), font, 0.5, config.COLORS['yellow'], 1)
                y_offset += 25
        
        # Draw risk timeline graph
        self._draw_risk_graph(frame, 650, 120, 500, 350)
        
        # Draw buttons
        new_btn, exit_btn = self._get_results_buttons()
        new_btn.is_hovered = new_btn.contains(self.mouse_x, self.mouse_y)
        exit_btn.is_hovered = exit_btn.contains(self.mouse_x, self.mouse_y)
        new_btn.draw(frame)
        exit_btn.draw(frame)
        
        # Footer
        footer = "Press 'R' for new exam, ESC to exit"
        foot_size = cv2.getTextSize(footer, font, 0.5, 1)[0]
        foot_x = (self.screen_width - foot_size[0]) // 2
        cv2.putText(frame, footer, (foot_x, self.screen_height - 30), 
                   font, 0.5, config.COLORS['light_gray'], 1)
        
        return frame
    
    def _draw_risk_graph(self, frame: np.ndarray, x: int, y: int, w: int, h: int):
        """Draw risk score timeline graph"""
        if self.session_results is None:
            return
        
        # Draw panel background
        self._draw_panel(frame, x, y, w, h, 0.5)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Title
        cv2.putText(frame, "RISK SCORE TIMELINE", (x + 20, y + 30), 
                   font, 0.7, config.COLORS['white'], 2)
        
        # Graph area
        graph_x = x + 60
        graph_y = y + 50
        graph_w = w - 80
        graph_h = h - 100
        
        # Draw axes
        cv2.line(frame, (graph_x, graph_y), (graph_x, graph_y + graph_h), 
                config.COLORS['white'], 2)
        cv2.line(frame, (graph_x, graph_y + graph_h), (graph_x + graph_w, graph_y + graph_h), 
                config.COLORS['white'], 2)
        
        # Y-axis labels (0, 50, 100)
        for val in [0, 50, 100]:
            label_y = graph_y + graph_h - int((val / 100) * graph_h)
            cv2.putText(frame, str(val), (graph_x - 35, label_y + 5), 
                       font, 0.4, config.COLORS['white'], 1)
            cv2.line(frame, (graph_x - 5, label_y), (graph_x, label_y), 
                    config.COLORS['white'], 1)
        
        # Axis labels
        cv2.putText(frame, "Score", (x + 10, y + 50 + graph_h // 2), 
                   font, 0.4, config.COLORS['white'], 1)
        cv2.putText(frame, "Time (seconds)", (graph_x + graph_w // 2 - 40, y + h - 10), 
                   font, 0.4, config.COLORS['white'], 1)
        
        # Get timeline data
        timeline = self.session_results.get('score_timeline', [])
        if len(timeline) < 2:
            cv2.putText(frame, "No events recorded", (graph_x + 50, graph_y + graph_h // 2), 
                       font, 0.6, config.COLORS['light_gray'], 1)
            return
        
        # Find max time
        max_time = max(t for t, s in timeline)
        if max_time <= 0:
            max_time = 1
        
        # Draw horizontal grid lines
        for val in [25, 50, 75]:
            grid_y = graph_y + graph_h - int((val / 100) * graph_h)
            cv2.line(frame, (graph_x, grid_y), (graph_x + graph_w, grid_y), 
                    (60, 60, 60), 1)
        
        # Draw step graph
        prev_point = None
        for timestamp, score in timeline:
            px = graph_x + int((timestamp / max_time) * graph_w)
            py = graph_y + graph_h - int((score / 100) * graph_h)
            
            if prev_point:
                # Horizontal line to current x
                cv2.line(frame, prev_point, (px, prev_point[1]), config.COLORS['cyan'], 2)
                # Vertical line to current y
                cv2.line(frame, (px, prev_point[1]), (px, py), config.COLORS['cyan'], 2)
            
            # Draw point marker
            cv2.circle(frame, (px, py), 4, config.COLORS['yellow'], -1)
            
            prev_point = (px, py)
        
        # Draw final horizontal line to end
        if prev_point:
            end_x = graph_x + graph_w
            cv2.line(frame, prev_point, (end_x, prev_point[1]), config.COLORS['cyan'], 2)
        
        # X-axis time labels
        duration = self.session_results['duration_seconds']
        for i in range(5):
            t = (i / 4) * duration
            label_x = graph_x + int((i / 4) * graph_w)
            cv2.putText(frame, f"{int(t)}s", (label_x - 10, graph_y + graph_h + 20), 
                       font, 0.35, config.COLORS['white'], 1)
    
    def _update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def _cleanup(self):
        """Cleanup resources"""
        print("\nShutting down...")
        
        if self.cap:
            self.cap.release()
        
        if self.pose_detector:
            self.pose_detector.release()
        
        cv2.destroyAllWindows()
        print("Goodbye!")


def main():
    """Main entry point"""
    print("=" * 60)
    print("    AI EXAM PROCTORING SYSTEM")
    print("    Cheating Pattern Detector")
    print("=" * 60)
    
    try:
        monitor = ExamMonitor()
        monitor.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
