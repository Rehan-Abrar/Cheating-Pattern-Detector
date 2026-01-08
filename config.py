"""
Configuration settings for the Cheating Pattern Detector
Adjustable thresholds and parameters for behavior detection
"""

# =============================================================================
# WEBCAM SETTINGS
# =============================================================================
CAMERA_INDEX = 0  # Default webcam
FRAME_WIDTH = 1280  # Higher resolution for full screen
FRAME_HEIGHT = 720

# =============================================================================
# HEAD ORIENTATION THRESHOLDS
# =============================================================================
HEAD_TURN_THRESHOLD = 25  # Degrees - head turn angle threshold
HEAD_TURN_DURATION = 1.0  # Seconds - duration to trigger suspicious behavior
GAZE_DEVIATION_THRESHOLD = 20  # Degrees - eye gaze deviation threshold

# =============================================================================
# FACE DETECTION THRESHOLDS
# =============================================================================
MIN_FACE_DETECTION_CONFIDENCE = 0.5
MULTIPLE_FACES_ALERT = True  # Alert when multiple faces detected
NO_FACE_DURATION = 1.5  # Seconds - duration of no face to trigger alert

# =============================================================================
# HAND DETECTION THRESHOLDS
# =============================================================================
HAND_VISIBILITY_THRESHOLD = 0.5  # Confidence threshold for hand visibility
HAND_MISSING_DURATION = 2.0  # Seconds - duration of missing hands to trigger alert

# =============================================================================
# RISK SCORING WEIGHTS (Cumulative - score only increases)
# =============================================================================
RISK_WEIGHTS = {
    'multiple_faces': 25,      # Another face detected
    'no_face': 20,             # No face in frame
    'hand_missing': 12,        # Hands not visible
    'head_turn': 10,           # Head turn detection
    'gaze_deviation': 8,       # Gaze deviation
    'looking_away': 10         # Combined head + gaze looking away
}

# =============================================================================
# EVENT COOLDOWNS (Minimum seconds between same event type)
# =============================================================================
EVENT_COOLDOWN = 3  # Seconds - cooldown between repeated events of same type

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================
SHOW_LANDMARKS = True
SHOW_RISK_SCORE = True
SHOW_EVENT_LOG = True
LOG_MAX_ENTRIES = 8  # Maximum event log entries to display on screen

# =============================================================================
# COLOR CODES (BGR format for OpenCV)
# =============================================================================
COLORS = {
    'green': (0, 255, 0),
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'blue': (255, 0, 0),
    'white': (255, 255, 255),
    'orange': (0, 165, 255),
    'dark_gray': (40, 40, 40),
    'light_gray': (200, 200, 200),
    'purple': (128, 0, 128),
    'cyan': (255, 255, 0)
}

# =============================================================================
# RISK LEVEL THRESHOLDS
# =============================================================================
RISK_LEVELS = {
    'low': (0, 31),       # 0-30
    'medium': (31, 61),   # 31-60
    'high': (61, 101)     # 61-100
}

# =============================================================================
# UI SETTINGS
# =============================================================================
BUTTON_COLOR = (0, 120, 0)        # Green button
BUTTON_HOVER_COLOR = (0, 180, 0)  # Lighter green on hover
BUTTON_TEXT_COLOR = (255, 255, 255)
STOP_BUTTON_COLOR = (0, 0, 180)   # Red button
STOP_BUTTON_HOVER = (0, 0, 220)   # Lighter red on hover
