"""
Risk Scorer Module
Calculates CUMULATIVE risk score based on detected suspicious behaviors
Score only increases when events occur - never decreases
"""

import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
import config


@dataclass
class RiskEvent:
    """Represents a risk-contributing event"""
    timestamp: float          # Seconds from exam start
    event_type: str           # Type of suspicious behavior
    risk_increment: int       # Points added to risk score
    total_score_after: int    # Cumulative score after this event


class RiskScorer:
    """
    Calculates CUMULATIVE risk score based on detected suspicious behaviors.
    
    Key Principles:
    - Score starts at 0
    - Score ONLY increases when suspicious events occur
    - Score NEVER decreases
    - Score is capped at 100
    """
    
    def __init__(self):
        # Current cumulative risk score (0-100)
        self.current_risk_score: int = 0
        
        # Session timing
        self.session_start: float = None
        self.session_end: float = None
        
        # Event history with timestamps
        self.event_log: List[RiskEvent] = []
        
        # Score history for graphing (timestamp, score)
        self.score_timeline: List[Tuple[float, int]] = []
        
        # Event counts by type
        self.event_counts: Dict[str, int] = {
            'multiple_faces': 0,
            'no_face': 0,
            'head_turn': 0,
            'gaze_deviation': 0,
            'looking_away': 0,
            'unauthorized_face': 0,
            'authorized_missing': 0
        }
        
        # Last event time by type (for cooldown)
        self.last_event_time: Dict[str, float] = {}
    
    def start_session(self):
        """Start a new monitoring session"""
        self.session_start = time.time()
        self.session_end = None
        self.current_risk_score = 0
        self.event_log.clear()
        self.score_timeline.clear()
        self.score_timeline.append((0.0, 0))  # Initial point at (0, 0)
        
        for key in self.event_counts:
            self.event_counts[key] = 0
        self.last_event_time.clear()
    
    def end_session(self):
        """End the monitoring session"""
        self.session_end = time.time()
    
    def add_event(self, event_type: str) -> bool:
        """
        Add a suspicious event and update the risk score.
        
        Args:
            event_type: Type of suspicious behavior detected
            
        Returns:
            True if event was recorded, False if on cooldown
        """
        if self.session_start is None:
            return False
        
        current_time = time.time()
        elapsed = current_time - self.session_start
        
        # Check cooldown
        if event_type in self.last_event_time:
            time_since_last = current_time - self.last_event_time[event_type]
            if time_since_last < config.EVENT_COOLDOWN:
                return False

        # Avoid double-counting: `looking_away` is a combined event
        # that includes head_turn + gaze_deviation. If a `looking_away`
        # event was recorded recently, skip separate head/gaze events
        # to prevent inflating the score.
        if event_type in ('head_turn', 'gaze_deviation'):
            last_lookaway = self.last_event_time.get('looking_away')
            if last_lookaway and (current_time - last_lookaway) < config.EVENT_COOLDOWN:
                return False
        
        # Get risk increment for this event type
        risk_increment = config.RISK_WEIGHTS.get(event_type, 0)
        
        if risk_increment == 0:
            return False
        
        # Update score (capped at 100)
        old_score = self.current_risk_score
        self.current_risk_score = min(100, self.current_risk_score + risk_increment)
        actual_increment = self.current_risk_score - old_score
        
        # Record event
        event = RiskEvent(
            timestamp=elapsed,
            event_type=event_type,
            risk_increment=actual_increment,
            total_score_after=self.current_risk_score
        )
        self.event_log.append(event)
        
        # Update timeline for graphing
        self.score_timeline.append((elapsed, self.current_risk_score))
        
        # Update counts and cooldown
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
        self.last_event_time[event_type] = current_time
        
        return True
    
    def get_current_score(self) -> int:
        """Get current cumulative risk score"""
        return self.current_risk_score
    
    def get_risk_level(self) -> str:
        """Get risk level based on current score"""
        score = self.current_risk_score
        for level, (low, high) in config.RISK_LEVELS.items():
            if low <= score < high:
                return level
        return 'high'
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since session start"""
        if self.session_start is None:
            return 0.0
        end_time = self.session_end if self.session_end else time.time()
        return end_time - self.session_start
    
    def get_formatted_time(self, seconds: float = None) -> str:
        """Format time as MM:SS"""
        if seconds is None:
            seconds = self.get_elapsed_time()
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def get_event_log_formatted(self) -> List[str]:
        """Get formatted event log for display"""
        formatted = []
        for event in self.event_log:
            time_str = self.get_formatted_time(event.timestamp)
            event_name = event.event_type.replace('_', ' ').title()
            formatted.append(f"{time_str} - {event_name} (+{event.risk_increment})")
        return formatted
    
    def get_recent_events(self, count: int = None) -> List[str]:
        """Get most recent events for display"""
        if count is None:
            count = config.LOG_MAX_ENTRIES
        formatted = self.get_event_log_formatted()
        return formatted[-count:] if len(formatted) > count else formatted
    
    def get_session_summary(self) -> Dict:
        """Get complete session summary for final results"""
        duration = self.get_elapsed_time()
        
        return {
            'final_score': self.current_risk_score,
            'risk_level': self.get_risk_level(),
            'duration_seconds': duration,
            'duration_formatted': self.get_formatted_time(duration),
            'total_events': len(self.event_log),
            'event_counts': self.event_counts.copy(),
            'event_log': self.event_log.copy(),
            'score_timeline': self.score_timeline.copy()
        }
    
    def get_score_timeline(self) -> List[Tuple[float, int]]:
        """Get score timeline for graphing"""
        return self.score_timeline.copy()
    
    def reset(self):
        """Reset scorer for new session"""
        self.current_risk_score = 0
        self.session_start = None
        self.session_end = None
        self.event_log.clear()
        self.score_timeline.clear()
        
        for key in self.event_counts:
            self.event_counts[key] = 0
        self.last_event_time.clear()
