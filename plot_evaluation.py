"""Generate evaluation visuals from the latest evaluation JSON in `results/`.

Produces:
 - results/confusion_matrix.png
 - results/metrics_bar.png

Usage:
  python plot_evaluation.py --results-dir results
"""
import os
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def find_latest_evaluation(results_dir: str):
    pats = glob.glob(os.path.join(results_dir, 'evaluation_*.json'))
    if not pats:
        return None
    return max(pats, key=os.path.getmtime)


def load_eval(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_confusion(tp, fp, fn, tn, out_path: str):
    # matrix: rows = actual, cols = predicted
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['pred_normal', 'pred_cheating'],
                yticklabels=['actual_normal', 'actual_cheating'])
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_metrics(precision, recall, f1, accuracy, out_path: str):
    labels = ['Precision', 'Recall', 'F1', 'Accuracy']
    vals = [precision, recall, f1, accuracy]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, vals, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
    plt.ylim(0, 1.02)
    plt.title('Model Performance Metrics')
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results')
    args = parser.parse_args()

    latest = find_latest_evaluation(args.results_dir)
    if latest is None:
        print('No evaluation_*.json found in', args.results_dir)
        return

    print('Loading', latest)
    ev = load_eval(latest)

    TP = int(ev.get('TP', 0))
    FP = int(ev.get('FP', 0))
    FN = int(ev.get('FN', 0))
    TN = int(ev.get('TN', 0))

    precision = float(ev.get('precision', 0.0))
    recall = float(ev.get('recall', 0.0))
    accuracy = float(ev.get('accuracy', 0.0))
    f1 = float(ev.get('f1', 0.0))

    os.makedirs(args.results_dir, exist_ok=True)
    cm_path = os.path.join(args.results_dir, 'confusion_matrix.png')
    metrics_path = os.path.join(args.results_dir, 'metrics_bar.png')

    plot_confusion(TP, FP, FN, TN, cm_path)
    plot_metrics(precision, recall, f1, accuracy, metrics_path)

    print('Saved:', cm_path)
    print('Saved:', metrics_path)


if __name__ == '__main__':
    main()
