# Cheating Pattern Detector
## AI-Based Online Exam Monitoring System

An intelligent monitoring system that detects suspicious behaviors during online examinations using computer vision and rule-based AI analysis. The system focuses on behavioral patterns rather than personal identity, promoting privacy-safe academic integrity.

---

## 📋 Project Overview

### Problem Statement
With the rapid growth of online education, conducting fair and secure online examinations has become a significant challenge. This project addresses the automatic detection of suspicious exam behavior using Artificial Intelligence, focusing on behavioral patterns rather than personal identity.

### Key Features
- **Real-time pose detection** using MediaPipe
- **Head orientation tracking** (yaw, pitch, roll angles)
- **Eye gaze direction analysis**
- **Multiple face detection**
- **Hand visibility monitoring**
- **Risk score calculation** (0-100%)
- **Privacy-safe** - No face recognition or identity storage

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam

### Setup Steps

1. **Clone or download the project**
   ```bash
   cd "d:\5th semester\AI\Cheating Pattern Detector"
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Running the Application
```bash
python main.py
```

### Controls
| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit application |
| `R` | Reset risk score and behavior counts |
| `L` | Toggle landmark display on/off |
| `S` | Show session summary in console |

---

## 📊 Detected Behaviors

| Behavior | Description | Weight |
|----------|-------------|--------|
| **Multiple Faces** | Additional person detected in frame | 25 |
| **No Face** | Student leaves the camera frame | 20 |
| **Hand Missing** | Both hands not visible for extended time | 12 |
| **Head Turn** | Student turns head away from screen (>25°) | 10 |
| **Looking Away** | Combined head turn + gaze deviation | 10 |
| **Gaze Deviation** | Eye gaze moves away from screen | 8 |

---

## 🏗️ System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Webcam Feed   │────▶│  Pose Detector   │────▶│    Behavior     │
│   (Input)       │     │  (MediaPipe)     │     │    Analyzer     │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐              │
                        │   Risk Scorer    │◀─────────────┘
                        │  (0-100% score)  │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │   Display UI     │
                        │  (OpenCV Window) │
                        └──────────────────┘
```

---

## 📁 Project Structure

```
Cheating Pattern Detector/
├── main.py              # Main application entry point
├── pose_detector.py     # MediaPipe-based landmark detection
├── behavior_analyzer.py # Rule-based behavior analysis
├── risk_scorer.py       # Risk score calculation
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## ⚙️ Configuration

Edit `config.py` to customize detection thresholds:

```python
# Head orientation thresholds
HEAD_TURN_THRESHOLD = 25      # Degrees
HEAD_TURN_DURATION = 1.0      # Seconds

# Face detection
NO_FACE_DURATION = 1.5        # Seconds

# Hand detection
HAND_MISSING_DURATION = 2.0   # Seconds

# Risk scoring weights
RISK_WEIGHTS = {
    'multiple_faces': 25,
    'no_face': 20,
    'hand_missing': 12,
    'head_turn': 10,
    'gaze_deviation': 8,
    'looking_away': 10
}
```

---

## 🎯 PEAS Analysis

| Component | Description |
|-----------|-------------|
| **Performance** | Accurate detection of suspicious behavior and reliable risk score |
| **Environment** | Partially observable, dynamic, real-time online exam environment |
| **Actuators** | Generate risk score and behavior logs |
| **Sensors** | Webcam video feed and extracted pose landmarks |

---

## 📈 Evaluation Metrics

Since this is a rule-based AI system (not trained on labeled data):

1. **Rule activation frequency** - How often each rule triggers
2. **Consistency of risk score** - Score stability across sessions
3. **Manual verification** - Comparing detected behaviors with observed behavior

---

## 🔮 Future Extensions

- [ ] Multi-student monitoring
- [ ] LMS integration
- [ ] Adaptive threshold learning
- [ ] Report generation
- [ ] Audio analysis (voice detection)

---

## 📄 License

This project is for educational purposes as part of an AI course.

---

## 👥 Authors

5th Semester AI Course Project
