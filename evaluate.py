"""Evaluate saved exam sessions for cheating detection performance.

Usage:
  - List labeled sessions and compute metrics:
      python evaluate.py --threshold 25

  - Label a session (add ground truth to a saved JSON):
      python evaluate.py --label session_20260111_192809.json --truth cheating

Scans the `results/` folder for session JSON files created by the app.
"""
import os
import json
import argparse
from typing import List, Dict
import config
import joblib
import numpy as np
from datetime import datetime


def find_result_files(results_dir: str) -> List[str]:
    if not os.path.isdir(results_dir):
        return []
    return [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.json')]


def load_session(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_session(path: str, data: Dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def label_session(path: str, truth: str):
    data = load_session(path)
    data['ground_truth'] = truth
    save_session(path, data)
    print(f"Labeled {os.path.basename(path)} as {truth}")


def evaluate_sessions(results_dir: str):
    files = find_result_files(results_dir)
    sessions = []
    for p in files:
        try:
            s = load_session(p)
            s['_path'] = p
            sessions.append(s)
        except Exception:
            print(f"Warning: failed to load {p}")

    labeled = [s for s in sessions if 'ground_truth' in s]
    print(f"Found {len(sessions)} session files, {len(labeled)} labeled with ground truth.")

    if len(labeled) == 0:
        print("No labeled sessions to evaluate. Use --label to add ground truth labels.")
        return

    TP = FP = FN = TN = 0
    rows = []

    # Try to load trained model (required)
    model_path = os.path.join(results_dir, 'model.pkl')
    meta_path = os.path.join(results_dir, 'model_meta.json')
    model = None
    feature_names = None
    if os.path.exists(model_path) and os.path.exists(meta_path):
        try:
            model = joblib.load(model_path)
            with open(meta_path, 'r', encoding='utf-8') as mf:
                meta = json.load(mf)
                feature_names = meta.get('feature_names')
            print(f'Loaded model from {model_path}')
        except Exception as e:
            print(f'Warning: failed to load model: {e}')
    else:
        print(f'No trained model found. Train one with: python train_model.py --results-dir {results_dir} --out {results_dir}')
        return

    def build_feature_vector(s: Dict, feature_names: List[str]):
        # Build a feature vector matching training feature order
        ev_counts = s.get('event_counts', {})
        vec = []
        # score timeline stats
        timeline = s.get('score_timeline', [])
        if len(timeline) < 2:
            scores = [pt[1] for pt in timeline] if timeline else []
            max_score = float(max(scores)) if scores else 0.0
            mean_inc = 0.0
            var_inc = 0.0
        else:
            scores = [pt[1] for pt in timeline]
            max_score = float(max(scores))
            inc = np.diff(scores)
            mean_inc = float(np.mean(inc)) if len(inc) > 0 else 0.0
            var_inc = float(np.var(inc)) if len(inc) > 0 else 0.0

        for fname in feature_names:
            if fname in ev_counts:
                vec.append(float(ev_counts.get(fname, 0)))
            elif fname == 'total_events':
                vec.append(float(s.get('total_events', 0)))
            elif fname == 'duration_seconds':
                vec.append(float(s.get('duration_seconds', 0.0)))
            elif fname == 'final_score':
                vec.append(float(s.get('final_score', 0.0)))
            elif fname == 'score_max':
                vec.append(float(max_score))
            elif fname == 'score_mean_increment':
                vec.append(float(mean_inc))
            elif fname == 'score_var_increment':
                vec.append(float(var_inc))
            else:
                # fallback
                vec.append(float(s.get(fname, 0.0)))
        return np.array(vec, dtype=float)
    for s in labeled:
        gt = s.get('ground_truth')
        score = s.get('final_score', None)
        if score is None:
            score = s.get('score', None)
        try:
            vec = build_feature_vector(s, feature_names)
            pred_raw = model.predict(vec.reshape(1, -1))
            pred = 'cheating' if int(pred_raw[0]) == 1 else 'normal'
        except Exception as e:
            print(f'Warning: model prediction failed for {s.get("_path")}: {e}')
            pred = 'normal'

        rows.append((s.get('session_id', os.path.basename(s['_path'])), gt, pred, score))

        if gt == 'cheating' and pred == 'cheating':
            TP += 1
        elif gt == 'cheating' and pred == 'normal':
            FN += 1
        elif gt == 'normal' and pred == 'cheating':
            FP += 1
        elif gt == 'normal' and pred == 'normal':
            TN += 1

    # Metrics (handle zero denominators)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\nConfusion Matrix:")
    print(f"TP: {TP}  FP: {FP}")
    print(f"FN: {FN}  TN: {TN}\n")

    print("Metrics:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"F1 Score:  {f1:.3f}\n")

    print("Detailed rows (session_id, ground_truth, prediction, final_score):")
    for r in rows:
        print(r)

    # Save evaluation summary
    summary = {
        'evaluated_at': datetime.now().isoformat(),
        'total_sessions': len(labeled),
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'precision': precision, 'recall': recall, 'accuracy': accuracy, 'f1': f1,
        'rows': rows
    }

    out_name = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path = os.path.join(results_dir, out_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Evaluation saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved exam sessions')
    parser.add_argument('--results-dir', default='results', help='Folder with session JSON files')
    # No threshold option: evaluation requires a trained model in results/
    parser.add_argument('--label', help='Path to session JSON to label (filename in results/)')
    parser.add_argument('--truth', choices=['cheating', 'normal'], help='Ground truth to attach when using --label')

    args = parser.parse_args()

    results_dir = args.results_dir

    if args.label:
        label_path = args.label
        # allow passing just filename
        if not os.path.isabs(label_path):
            label_path = os.path.join(results_dir, label_path)
        if not os.path.exists(label_path):
            print(f"File not found: {label_path}")
            return
        if not args.truth:
            print("Please specify --truth cheating|normal when labeling")
            return
        label_session(label_path, args.truth)
        return

    evaluate_sessions(results_dir)


if __name__ == '__main__':
    main()
