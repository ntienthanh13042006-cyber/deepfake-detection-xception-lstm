import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix
)

from dataset import DeepfakeDataset
from model import SpatialTemporalXceptionBiLSTM


# ============================================================
# 1. CẤU HÌNH
# ============================================================

DATA_ROOT = r"D:\Project\Processed_Data_v2"

VAL_CSV = r"D:\Project\splits\val.csv"
TEST_CSV = r"D:\Project\splits\test.csv"

CHECKPOINT_PATH = (
    r"D:\Project\checkpoints\best_model.pth"
)

RESULTS_DIR = r"D:\Project\results"

BATCH_SIZE = 2

NUM_WORKERS = 0

SEQUENCE_LENGTH = 15

LSTM_HIDDEN_SIZE = 256

LSTM_LAYERS = 2

# Ngưỡng mặc định để so sánh
BASELINE_THRESHOLD = 0.50

# Quét từ 0.10 đến 0.90
THRESHOLD_START = 0.10
THRESHOLD_END = 0.90
THRESHOLD_STEP = 0.01


# ============================================================
# 2. DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print("\n[INFO] CUDA khả dụng")

        print(
            f"[INFO] GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        device = torch.device("cpu")

        print(
            "\n[WARNING] CUDA không khả dụng."
        )

        print("[INFO] Sử dụng CPU.")

    return device


# ============================================================
# 3. LOAD MODEL
# ============================================================

def load_model(device):

    if not os.path.exists(CHECKPOINT_PATH):

        raise FileNotFoundError(
            f"Không tìm thấy checkpoint:\n"
            f"{CHECKPOINT_PATH}"
        )

    print("\n========== LOAD MODEL ==========")

    model = SpatialTemporalXceptionBiLSTM(
        sequence_length=SEQUENCE_LENGTH,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    if "model_state_dict" not in checkpoint:

        raise KeyError(
            "Checkpoint không chứa "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    print(
        f"[INFO] Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'N/A')}"
    )

    print(
        f"[INFO] Validation ROC-AUC: "
        f"{checkpoint.get('val_auc', 'N/A'):.4f}"
    )

    print("[INFO] Model loaded thành công.")

    return model, checkpoint


# ============================================================
# 4. LẤY PROBABILITY TỪ DATASET
# ============================================================

@torch.no_grad()
def get_predictions(
    model,
    loader,
    device
):

    model.eval()

    all_labels = []

    all_probabilities = []

    for batch_idx, (inputs, labels) in enumerate(loader):

        inputs = inputs.to(
            device,
            non_blocking=True
        )

        logits = model(inputs)

        logits = logits.squeeze(1)

        probabilities = torch.sigmoid(
            logits
        )

        all_labels.extend(
            labels.numpy().astype(int)
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        if (batch_idx + 1) % 20 == 0:

            print(
                f"      Batch "
                f"{batch_idx + 1}/"
                f"{len(loader)}"
            )

    return (
        np.asarray(all_labels),
        np.asarray(all_probabilities)
    )


# ============================================================
# 5. TÍNH METRICS THEO THRESHOLD
# ============================================================

def calculate_threshold_metrics(
    labels,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        labels,
        predictions
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "balanced_accuracy":
            balanced_acc
    }


# ============================================================
# 6. QUÉT THRESHOLD TRÊN VALIDATION
# ============================================================

def search_best_threshold(
    labels,
    probabilities
):

    print(
        "\n========== THRESHOLD ANALYSIS =========="
    )

    thresholds = np.arange(
        THRESHOLD_START,
        THRESHOLD_END +
        THRESHOLD_STEP / 2,
        THRESHOLD_STEP
    )

    results = []

    for threshold in thresholds:

        metrics = calculate_threshold_metrics(
            labels,
            probabilities,
            float(threshold)
        )

        results.append(metrics)

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Chọn theo Balanced Accuracy
    # --------------------------------------------------------

    results_sorted = results_df.sort_values(
        by=[
            "balanced_accuracy",
            "macro_f1"
        ],
        ascending=[
            False,
            False
        ]
    ).reset_index(
        drop=True
    )

    best = results_sorted.iloc[0]

    best_threshold = float(
        best["threshold"]
    )

    print("\n[INFO] BEST THRESHOLD")

    print(
        f"Threshold:        "
        f"{best_threshold:.2f}"
    )

    print(
        f"Accuracy:         "
        f"{best['accuracy']:.4f}"
    )

    print(
        f"Precision:        "
        f"{best['precision']:.4f}"
    )

    print(
        f"Recall:           "
        f"{best['recall']:.4f}"
    )

    print(
        f"F1:               "
        f"{best['f1']:.4f}"
    )

    print(
        f"Macro-F1:         "
        f"{best['macro_f1']:.4f}"
    )

    print(
        f"Balanced Accuracy:"
        f" {best['balanced_accuracy']:.4f}"
    )

    # --------------------------------------------------------
    # Lưu toàn bộ kết quả
    # --------------------------------------------------------

    threshold_csv = os.path.join(
        RESULTS_DIR,
        "validation_threshold_results.csv"
    )

    results_df.to_csv(
        threshold_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\n[INFO] Saved: {threshold_csv}"
    )

    # --------------------------------------------------------
    # Lưu threshold tối ưu
    # --------------------------------------------------------

    threshold_json = os.path.join(
        RESULTS_DIR,
        "optimal_threshold.json"
    )

    threshold_config = {
        "selection_dataset": "Validation",
        "selection_metric":
            "Balanced Accuracy",
        "optimal_threshold":
            best_threshold,

        "validation_metrics": {
            "accuracy":
                float(best["accuracy"]),
            "precision":
                float(best["precision"]),
            "recall":
                float(best["recall"]),
            "f1":
                float(best["f1"]),
            "macro_f1":
                float(best["macro_f1"]),
            "balanced_accuracy":
                float(
                    best["balanced_accuracy"]
                )
        }
    }

    with open(
        threshold_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            threshold_config,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(
        f"[INFO] Saved: {threshold_json}"
    )

    # --------------------------------------------------------
    # Vẽ biểu đồ
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        results_df["threshold"],
        results_df["balanced_accuracy"],
        label="Balanced Accuracy",
        linewidth=2
    )

    plt.plot(
        results_df["threshold"],
        results_df["macro_f1"],
        label="Macro F1",
        linewidth=2
    )

    plt.plot(
        results_df["threshold"],
        results_df["f1"],
        label="Fake F1",
        linewidth=2
    )

    plt.axvline(
        best_threshold,
        linestyle="--",
        linewidth=2,
        label=(
            f"Best threshold = "
            f"{best_threshold:.2f}"
        )
    )

    plt.xlabel(
        "Decision Threshold"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Threshold Analysis on Validation Set"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plot_path = os.path.join(
        RESULTS_DIR,
        "validation_threshold_analysis.png"
    )

    plt.savefig(
        plot_path,
        dpi=300
    )

    plt.close()

    print(
        f"[INFO] Saved: {plot_path}"
    )

    return best_threshold


# ============================================================
# 7. EVALUATE TEST VỚI THRESHOLD ĐÃ KHÓA
# ============================================================

def evaluate_test(
    labels,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "threshold": threshold,

        "accuracy": accuracy_score(
            labels,
            predictions
        ),

        "precision": precision_score(
            labels,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            labels,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            labels,
            predictions,
            zero_division=0
        ),

        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        ),

        "balanced_accuracy":
            balanced_accuracy_score(
                labels,
                predictions
            ),

        "roc_auc":
            roc_auc_score(
                labels,
                probabilities
            )
    }

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    )

    return metrics, cm, predictions


# ============================================================
# 8. IN TEST RESULTS
# ============================================================

def print_test_results(
    metrics,
    cm
):

    print("\n")

    print("=" * 65)

    print(
        "       FINAL TEST - OPTIMAL THRESHOLD"
    )

    print("=" * 65)

    print(
        f"Threshold:           "
        f"{metrics['threshold']:.2f}"
    )

    print(
        f"Accuracy:            "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision:           "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:              "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1-score:            "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Macro-F1:            "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Balanced Accuracy:   "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"ROC-AUC:             "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        "\n========== CONFUSION MATRIX =========="
    )

    print(
        "                 Predicted"
    )

    print(
        "              Real    Fake"
    )

    print(
        f"Actual Real   "
        f"{cm[0,0]:5d}   {cm[0,1]:5d}"
    )

    print(
        f"Actual Fake   "
        f"{cm[1,0]:5d}   {cm[1,1]:5d}"
    )


# ============================================================
# 9. LƯU TEST RESULT
# ============================================================

def save_test_results(
    metrics,
    cm
):

    output = {
        "threshold_selection": {
            "dataset": "Validation",
            "metric": "Balanced Accuracy"
        },

        "test": {
            "threshold":
                float(metrics["threshold"]),
            "accuracy":
                float(metrics["accuracy"]),
            "precision":
                float(metrics["precision"]),
            "recall":
                float(metrics["recall"]),
            "f1":
                float(metrics["f1"]),
            "macro_f1":
                float(metrics["macro_f1"]),
            "balanced_accuracy":
                float(metrics["balanced_accuracy"]),
            "roc_auc":
                float(metrics["roc_auc"])
        },

        "confusion_matrix": [
            cm[0].tolist(),
            cm[1].tolist()
        ]
    }

    output_path = os.path.join(
        RESULTS_DIR,
        "test_optimal_threshold_summary.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(
        f"\n[INFO] Saved: {output_path}"
    )


# ============================================================
# 10. SO SÁNH THRESHOLD 0.50 VS OPTIMAL
# ============================================================

def compare_thresholds(
    labels,
    probabilities,
    optimal_threshold
):

    baseline = calculate_threshold_metrics(
        labels,
        probabilities,
        BASELINE_THRESHOLD
    )

    optimal = calculate_threshold_metrics(
        labels,
        probabilities,
        optimal_threshold
    )

    comparison = pd.DataFrame(
        [
            {
                "setting": "Baseline",
                **baseline
            },
            {
                "setting": "Optimal",
                **optimal
            }
        ]
    )

    output_path = os.path.join(
        RESULTS_DIR,
        "threshold_comparison_validation.csv"
    )

    comparison.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\n========== VALIDATION COMPARISON =========="
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        f"\n[INFO] Saved: {output_path}"
    )


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 65)

    print(
        "       DEEPFAKE DETECTION"
    )

    print(
        "       THRESHOLD ANALYSIS"
    )

    print("=" * 65)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    device = get_device()

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model, checkpoint = load_model(
        device
    )

    # ========================================================
    # VALIDATION DATASET
    # ========================================================

    print(
        "\n========== LOAD VALIDATION DATASET =========="
    )

    val_dataset = DeepfakeDataset(
        csv_path=VAL_CSV,
        data_root=DATA_ROOT
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    print(
        f"[INFO] Validation samples: "
        f"{len(val_dataset)}"
    )

    print(
        f"[INFO] Validation batches: "
        f"{len(val_loader)}"
    )

    # ========================================================
    # GET VALIDATION PROBABILITIES
    # ========================================================

    print(
        "\n========== VALIDATION INFERENCE =========="
    )

    val_labels, val_probabilities = get_predictions(
        model=model,
        loader=val_loader,
        device=device
    )

    # ========================================================
    # SEARCH OPTIMAL THRESHOLD
    # ========================================================

    optimal_threshold = search_best_threshold(
        labels=val_labels,
        probabilities=val_probabilities
    )

    # ========================================================
    # VALIDATION COMPARISON
    # ========================================================

    compare_thresholds(
        labels=val_labels,
        probabilities=val_probabilities,
        optimal_threshold=optimal_threshold
    )

    # ========================================================
    # TEST DATASET
    # ========================================================

    print(
        "\n========== LOAD TEST DATASET =========="
    )

    test_dataset = DeepfakeDataset(
        csv_path=TEST_CSV,
        data_root=DATA_ROOT
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    print(
        f"[INFO] Test samples: "
        f"{len(test_dataset)}"
    )

    print(
        f"[INFO] Test batches: "
        f"{len(test_loader)}"
    )

    # ========================================================
    # TEST
    # ========================================================

    print(
        "\n========== TEST INFERENCE =========="
    )

    test_labels, test_probabilities = get_predictions(
        model=model,
        loader=test_loader,
        device=device
    )

    # ========================================================
    # APPLY LOCKED THRESHOLD
    # ========================================================

    test_metrics, cm, test_predictions = evaluate_test(
        labels=test_labels,
        probabilities=test_probabilities,
        threshold=optimal_threshold
    )

    print_test_results(
        metrics=test_metrics,
        cm=cm
    )

    save_test_results(
        metrics=test_metrics,
        cm=cm
    )

    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    predictions_df = pd.DataFrame(
        {
            "true_label":
                test_labels,

            "probability_fake":
                test_probabilities,

            "predicted_label":
                test_predictions
        }
    )

    predictions_path = os.path.join(
        RESULTS_DIR,
        "test_predictions_optimal_threshold.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"[INFO] Saved: {predictions_path}"
    )

    # ========================================================
    # END
    # ========================================================

    print("\n")

    print("=" * 65)

    print(
        "       THRESHOLD ANALYSIS COMPLETED"
    )

    print("=" * 65)

    print(
        f"[INFO] Optimal threshold: "
        f"{optimal_threshold:.2f}"
    )

    print(
        f"[INFO] Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'N/A')}"
    )


if __name__ == "__main__":
    main()