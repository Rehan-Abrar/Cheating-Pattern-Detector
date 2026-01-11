"""Train a supervised classifier on labeled session JSONs in `results/`.

Produces a saved model at `results/model.pkl` and metadata `results/model_meta.json`.

Usage:
  python train_model.py --results-dir results --out results
"""
import os
import json
import argparse
from typing import List, Dict, Tuple
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib


FEATURE_KEYS = [
    # event counts will be appended dynamically
    'total_events',
    'duration_seconds',
    'final_score',
    'score_max',
    'score_mean_increment',
    'score_var_increment'
]


def find_sessions(results_dir: str) -> List[str]:
    if not os.path.isdir(results_dir):
        return []
    return [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.json')]


def load_session(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_features(s: Dict, event_names: List[str]) -> List[float]:
    # event counts
    ev_counts = s.get('event_counts', {})
    features = [float(ev_counts.get(k, 0)) for k in event_names]

    # totals and durations
    features.append(float(s.get('total_events', 0)))
    features.append(float(s.get('duration_seconds', 0.0)))

    # final score
    features.append(float(s.get('final_score', 0.0)))

    # score timeline stats: compute increments between successive timeline points
    timeline = s.get('score_timeline', [])
    if len(timeline) < 2:
        max_score = timeline[-1][1] if timeline else 0.0
        mean_inc = 0.0
        var_inc = 0.0
    else:
        scores = [pt[1] for pt in timeline]
        max_score = float(max(scores))
        increments = np.diff(scores)
        mean_inc = float(np.mean(increments)) if len(increments) > 0 else 0.0
        var_inc = float(np.var(increments)) if len(increments) > 0 else 0.0

    features.append(max_score)
    features.append(mean_inc)
    features.append(var_inc)

    return features


def build_dataset(files: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
    sessions = []
    event_names = None
    X = []
    y = []
    names = []

    for p in files:
        try:
            s = load_session(p)
        except Exception:
            continue

        gt = s.get('ground_truth')
        if gt not in ('cheating', 'normal'):
            continue

        ev_counts = s.get('event_counts', {})
        if event_names is None:
            event_names = sorted(list(ev_counts.keys()))

        features = extract_features(s, event_names)
        X.append(features)
        y.append(1 if gt == 'cheating' else 0)
        names.append(os.path.basename(p))

    if event_names is None:
        event_names = []

    feature_names = event_names + FEATURE_KEYS

    return np.array(X, dtype=float), np.array(y, dtype=int), feature_names, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--out', default='results')
    args = parser.parse_args()

    files = find_sessions(args.results_dir)
    X, y, feature_names, names = build_dataset(files)

    if X.size == 0:
        print('No labeled sessions found. Label sessions first using evaluate.py --label')
        return

    print(f'Loaded {X.shape[0]} sessions with {X.shape[1]} features')

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train Logistic Regression only
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    # Evaluate
    print('\nLogistic Regression:')
    print(classification_report(y_test, y_pred, target_names=['normal', 'cheating']))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))

    best_model = lr
    best_name = 'LogisticRegression'

    # Save model and metadata
    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, 'model.pkl')
    meta_path = os.path.join(args.out, 'model_meta.json')
    joblib.dump(best_model, model_path)

    meta = {
        'feature_names': feature_names,
        'model_type': best_name,
        'n_samples': int(X.shape[0])
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()
