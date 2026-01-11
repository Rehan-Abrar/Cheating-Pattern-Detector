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
- **Risk score calculation** (0-100%)
- **Privacy-safe** - No face recognition or identity storage

---

## 🗂️ Dataset Description

### Dataset Source
- Self-collected sessions during trial exams.

Each session consists of video recordings of the student performing exam tasks.

Planned/target per-exam recording: 5–60 minutes (to simulate full exam conditions).

Optionally, augmented or simulated datasets may be used for training under varied conditions.

### Dataset Type
- Video data: Frames processed for landmarks and embeddings (pose/face features).

- Numerical features: Extracted per-session aggregates, including event counts, gaze deviations, head turns, and risk scores, used for model training.

Note: Raw frames are not stored for training unless explicitly required; only extracted features are used.

### Key Features
- Video-derived features: Face landmarks, iris positions, head yaw/pitch.

- Event-based features: `head_turn`, `gaze_deviation`, `looking_away`, `no_face`, `multiple_faces`, `authorized_missing`, `unauthorized_face`.

- Score-based features: Cumulative risk score, timeline increments, final score.

- Aggregates: Event counts, duration, max/mean/variance of score increments per session — these are used as input features for the supervised model.

### Dataset Size
- Current dataset: 19 labeled sessions in the repository.

- Each session is short (~11–79 seconds), sampled at 3 fps → ~33–237 frames per session.

- Target dataset: 100+ sessions per class (normal and cheating) for robust supervised learning.

- Planned exam recordings: 5–60 minutes, sampled at 3 fps → ~900–10,800 frames per session.

### Labeling
- All session labels (normal vs cheating) are human-provided to prevent label leakage and ensure reliable supervised training.

### Privacy Considerations
- Face embeddings (`faces/authorized.npy`) and session data are sensitive.
- Consent from participants is required for recording.
- Store data securely with limited access and follow institutional privacy guidelines.


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

# Cheating Pattern Detector

AI-Based Online Exam Monitoring — implementation, training and evaluation

This repository implements a real-time exam proctoring prototype that:
- detects suspicious behaviors (head turns, gaze deviation, missing/unauthorized faces),
- verifies the enrolled student using 1:1 face recognition (enrollment + verification),
- saves session summaries for offline analysis and supervised training, and
- supports training/evaluation of a supervised classifier (Logistic Regression) from labeled sessions.

## Contents
- `main.py` — app entry (OpenCV UI, enrollment, monitoring loop)
- `pose_detector.py` — MediaPipe-based landmarks, iris & head-angle estimators
- `face_recognizer.py` — enrollment and verification (dlib/face_recognition)
- `behavior_analyzer.py` — rules turning landmarks into events
- `risk_scorer.py` — event weights, cooldowns, session summary builder
- `train_model.py` — feature extraction and Logistic Regression training
- `evaluate.py` — model-based evaluation (requires `results/model.pkl`)
- `plot_evaluation.py` — confusion matrix + metrics bar plot generation
- `results/` — session JSONs, trained `model.pkl`, evaluation JSONs and plots
- `faces/` — `authorized.npy` (enrolled face embedding)
- `requirements.txt` — Python dependencies

## Quick status
- Current dataset included: 19 labeled sessions
- Current supervised model: Logistic Regression (saved to `results/model.pkl`)

## Installation
1. Create and activate a virtual environment (Windows PowerShell):
```powershell
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
pip install -r requirements.txt
```
2. Notes for Windows: `face_recognition`/`dlib` may require CMake and Visual Studio Build Tools. MediaPipe wheels must expose `mp.solutions` on some Windows Python versions — use the provided `requirements.txt` and follow installation errors.

## Running the app
```powershell
python main.py
```

### UI keys
- `R`: reset score
- `S`: show session summary in console
- `Q` / `ESC`: quit

## Enrollment (1:1 face verification)
- Press the on-screen/register control (or the key bound in `main.py`) to enter `STATE_REGISTER`.
- The app collects ~20 face crops, computes `face_recognition` embeddings per capture, averages them, and saves `faces/authorized.npy` (single-vector enrollment).
- Enrollment should be performed under exam-like lighting and camera position for best results.

## Monitoring loop (what happens per frame)
1. Capture frame (OpenCV)
2. Run MediaPipe face/pose mesh in `pose_detector.py` to extract landmarks and iris centers
3. Run `face_recognizer.py` to verify face embedding (authorized/unauthorized/missing)
4. `behavior_analyzer.py` converts landmark measurements into high-level events (`head_turn`, `gaze_deviation`, `looking_away`, `no_face`, `multiple_faces`, `authorized_missing`, `unauthorized_face`)
5. `risk_scorer.py` applies per-event weights, cooldowns, caps the risk at 100, and records an event timeline
6. App overlays debug visuals (landmarks, iris centers, gaze angle) and displays the current risk score
7. When monitoring stops, `get_session_summary()` is saved to `results/session_YYYYMMDD_HHMMSS.json`

## Behavior detection details
- Gaze and head orientation: computed from face landmarks and iris centers; thresholds live in `config.py` (`HEAD_TURN_THRESHOLD`, `GAZE_DEVIATION_THRESHOLD`).
- `looking_away` is a composed event to avoid double counting immediate head/gaze sub-events; scorer cooldowns prevent rapid double-counting.

## Face recognition
- Enrollment: average of 20 embeddings to reduce noise; saved to `faces/authorized.npy`.
- Verification: compute embedding per detected face and compare Euclidean distance against a configured threshold — unauthorized => event weight (configured in `config.py`).

## Session persistence
- Each session JSON contains `final_score`, `risk_level`, `duration_seconds`, `total_events`, `event_counts`, `event_log` (timestamped events), `score_timeline`, `saved_at`, and `ground_truth` (if labeled).

## Supervised training (`train_model.py`)
- Scans `results/` for human-labeled sessions (`ground_truth` ∈ {`cheating`, `normal`}).
- Extracts per-session features: event counts, `total_events`, `duration_seconds`, `final_score`, `score_max`, `score_mean_increment`, `score_var_increment`.
- Trains a `LogisticRegression` model (baseline, interpretable coefficients). Model artifact saved as `results/model.pkl` and metadata `results/model_meta.json` (feature order, model type, sample count).

## Evaluation (`evaluate.py`)
- Requires `results/model.pkl` and `results/model_meta.json`.
- Builds feature vectors in the same order as saved metadata and uses the trained model to predict `normal` vs `cheating` for each labeled session.
- Computes TP/FP/FN/TN, Precision, Recall, Accuracy, F1 and writes `results/evaluation_YYYYMMDD_HHMMSS.json`.

## Visuals (`plot_evaluation.py`)
- Produces `results/confusion_matrix.png` and `results/metrics_bar.png` from the latest evaluation JSON.

## Dataset (summary)
- Current dataset: 19 labeled sessions included in `results/`.
- Session lengths in repo: short (approx. 11–79 seconds). Sampling rate used by the pipeline: 3 fps — thus existing sessions at 3 fps → roughly 33–237 frames/session.
- Planned exam recordings (target): 5–60 minutes sampled at 3 fps → ~900–10,800 frames/session.
- Target dataset size for robust supervised models: 100+ sessions per class (normal and cheating).

## Privacy & ethics
- Enrollment embeddings (`faces/authorized.npy`) and session JSONs are sensitive. Obtain explicit consent before recording.
- Limit retention, restrict access, and consider encrypting stored embeddings.
- This system detects observable behaviors — do not infer intent; always include human review for high-stakes decisions.

## Reproducibility & notes
- `requirements.txt` contains the Python dependencies used; install into a virtual environment as shown above.
- Training/evaluation are deterministic if seeds are fixed; `train_model.py` saves metadata including feature order and sample counts.

## Troubleshooting & platform notes
- If `face_recognition`/`dlib` fails to install on Windows, install CMake and Build Tools for Visual Studio, or use prebuilt wheels.
- MediaPipe version compatibility: on Windows some pip wheels may not expose `mp.solutions`; use a wheel that matches your Python version.

## Next steps (recommended)
- Collect more labeled sessions and re-run `train_model.py` with stratified cross-validation and hyperparameter tuning.
- Add a small interactive labeler to confirm `ground_truth` for each session.
- Add unit tests for feature extraction and evaluator logic.
- Secure storage for `faces/authorized.npy` if deploying beyond testing.

## Contact
- Project for a 5th semester AI course. For questions, inspect code or open an issue in the repository.

---

## 📄 License

This project is for educational purposes as part of an AI course.

---

## 👥 Authors

5th Semester AI Course Project
