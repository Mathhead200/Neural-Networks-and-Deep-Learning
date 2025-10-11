"""
Simple K-Nearest Neighbors classifier experiment on MNIST CSV files.

This script expects two CSV files in the same directory:
 - mnist_train.csv  (60,000 rows, label + 784 pixels)
 - mnist_test.csv   (10,000 rows, label + 784 pixels)

It trains KNeighborsClassifier for a range of k values, measures
accuracy and prediction time, and saves a small plot and CSV of results.

Usage examples:
  python a.py                      # runs default ks [1,3,5,7,9,11]
  python a.py --ks 1,3,5,7,9
  python a.py --ks 1-15:2 --train-sample 5000

Notes:
 - Pixel values are scaled to [0,1] by dividing by 255.
 - You can reduce --train-sample to speed up experiments on slower machines.
"""

import argparse
import time
import csv
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def parse_ks(ks_arg: str) -> List[int]:
    """Parse a ks argument which can be comma-separated or a range like start-end:step.

    Examples:
      "1,3,5"
      "1-15"        -> equivalent to 1-15 step 1
      "1-15:2"      -> 1,3,5,...,15
    """
    if not ks_arg:
        return [1, 3, 5, 7, 9, 11]
    ks_arg = ks_arg.strip()
    if "," in ks_arg:
        return [int(x) for x in ks_arg.split(",") if x.strip()]
    if "-" in ks_arg:
        # support optional step "start-end:step"
        range_part = ks_arg
        step = 1
        if ":" in ks_arg:
            range_part, step_part = ks_arg.split(":", 1)
            step = int(step_part)
        start_s, end_s = range_part.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        return list(range(start, end + 1, step))
    # single integer
    return [int(ks_arg)]


def load_mnist_csv(path: str):
    """Load MNIST CSV and return (X, y).

    This handles CSVs with or without a header and where the label is
    either named 'label' or is the first column.
    """
    df = pd.read_csv(path)
    # determine label column
    if "label" in df.columns:
        y = df["label"].values
        X = df.drop(columns=["label"]).values
    else:
        # assume first column is label
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
    return X, y


def run_experiment(
    train_csv: str,
    test_csv: str,
    ks: List[int],
    train_sample: Optional[int] = None,
    out_csv: str = "knn_results.csv",
    out_plot: str = "knn_tradeoff.png",
):
    # Load data
    print(f"Loading training data from {train_csv}...")
    X_train, y_train = load_mnist_csv(train_csv)
    print(f"Loading test data from {test_csv}...")
    X_test, y_test = load_mnist_csv(test_csv)

    # Optionally subsample training set to speed up experiments
    if train_sample is not None and train_sample > 0 and train_sample < X_train.shape[0]:
        print(f"Downsampling training set to {train_sample} examples (random)...")
        rng = np.random.default_rng(42)
        idx = rng.choice(X_train.shape[0], size=train_sample, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    # Scale pixels to [0,1]
    print("Scaling pixel values to [0,1] by dividing by 255...")
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    results = []

    for k in ks:
        print(f"\nTraining KNeighborsClassifier with k={k}...")
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

        # Fit (this stores the training points; k-NN has little to no "training" cost)
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        t1 = time.perf_counter()
        fit_time = t1 - t0

        # Predict and time the predictions
        print("Predicting test set...")
        t0 = time.perf_counter()
        y_pred = clf.predict(X_test)
        t1 = time.perf_counter()
        predict_time = t1 - t0

        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        print(f"k={k}: accuracy={acc:.4f}, fit_time={fit_time:.3f}s, predict_time={predict_time:.3f}s")

        results.append({"k": k, "accuracy": acc, "fit_time": fit_time, "predict_time": predict_time})

    # Save results to CSV
    print(f"Saving results to {out_csv}...")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "accuracy", "fit_time", "predict_time"]) 
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Plot accuracy and prediction time vs k
    ks_vals = [r["k"] for r in results]
    acc_vals = [r["accuracy"] for r in results]
    t_vals = [r["predict_time"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ks_vals, acc_vals, marker="o", color="tab:blue", label="accuracy")
    ax1.set_xlabel("k (n_neighbors)")
    ax1.set_ylabel("Accuracy", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, which="both", axis="y", linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ks_vals, t_vals, marker="x", color="tab:red", label="predict_time (s)")
    ax2.set_ylabel("Prediction time (s)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Title and legend
    fig.suptitle("k-NN trade-off: accuracy vs prediction time")
    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print(f"Saving plot to {out_plot}...")
    plt.savefig(out_plot, dpi=150)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="k-NN experiments on MNIST CSV files")
    parser.add_argument("--train", default="mnist_train.csv", help="Path to train CSV (default: mnist_train.csv)")
    parser.add_argument("--test", default="mnist_test.csv", help="Path to test CSV (default: mnist_test.csv)")
    parser.add_argument("--ks", default="1,3,5,7,9,11", help="Comma list or range for k (e.g. '1,3,5' or '1-11:2')")
    parser.add_argument("--train-sample", type=int, default=None, help="Optional: number of training samples to randomly subsample (speeds up k-NN)")
    parser.add_argument("--out-csv", default="knn_results.csv", help="CSV file to write results to")
    parser.add_argument("--out-plot", default="knn_tradeoff.png", help="Output plot filename")

    args = parser.parse_args()

    ks = parse_ks(args.ks)

    run_experiment(
        train_csv=args.train,
        test_csv=args.test,
        ks=ks,
        train_sample=args.train_sample,
        out_csv=args.out_csv,
        out_plot=args.out_plot,
    )


if __name__ == "__main__":
    main()
