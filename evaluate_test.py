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
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from dataset import DeepfakeDataset
from model import SpatialTemporalXceptionBiLSTM


# ============================================================
# 1. CẤU HÌNH
# ============================================================

DATA_ROOT = r"D:\Project\Processed_Data_v2"

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

THRESHOLD = 0.5


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

    print(
        "\n========== LOAD MODEL =========="
    )

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
        f"{checkpoint.get('val_auc', 'N/A')}"
    )

    print(
        "[INFO] Model loaded thành công."
    )

    return model, checkpoint


# ============================================================
# 4. EVALUATE TEST
# ============================================================

@torch.no_grad()
def evaluate_test(
    model,
    dataset,
    loader,
    device
):

    model.eval()

    all_labels = []

    all_probabilities = []

    all_predictions = []

    all_paths = []

    all_methods = []

    current_index = 0

    print(
        "\n========== ĐÁNH GIÁ TEST =========="
    )

    for batch_index, (inputs, labels) in enumerate(
        loader
    ):

        inputs = inputs.to(
            device,
            non_blocking=True
        )

        logits = model(inputs)

        logits = logits.squeeze(1)

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= THRESHOLD
        ).long()

        batch_size = inputs.size(0)

        # -----------------------------
        # Labels
        # -----------------------------

        batch_labels = (
            labels.numpy()
            .astype(int)
            .tolist()
        )

        batch_probs = (
            probabilities.cpu()
            .numpy()
            .tolist()
        )

        batch_predictions = (
            predictions.cpu()
            .numpy()
            .astype(int)
            .tolist()
        )

        all_labels.extend(
            batch_labels
        )

        all_probabilities.extend(
            batch_probs
        )

        all_predictions.extend(
            batch_predictions
        )

        # -----------------------------
        # Metadata từ CSV
        # -----------------------------

        batch_rows = dataset.df.iloc[
            current_index:
            current_index + batch_size
        ]

        all_paths.extend(
            batch_rows[
                "relative_path"
            ].tolist()
        )

        all_methods.extend(
            batch_rows[
                "method"
            ].tolist()
        )

        current_index += batch_size

        if (batch_index + 1) % 10 == 0:

            print(
                f"      Batch "
                f"{batch_index + 1}/"
                f"{len(loader)}"
            )

    return (
        np.array(all_labels),
        np.array(all_probabilities),
        np.array(all_predictions),
        all_paths,
        all_methods
    )


# ============================================================
# 5. TÍNH METRICS
# ============================================================

def calculate_metrics(
    labels,
    predictions,
    probabilities
):

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

    auc_score = roc_auc_score(
        labels,
        probabilities
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc_score)
    }


# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    labels,
    predictions
):

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
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

    # -----------------------------
    # Vẽ confusion matrix
    # -----------------------------

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(cm)

    ax.set_title(
        "Confusion Matrix - Test Set"
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "True Label"
    )

    ax.set_xticks(
        [0, 1],
        ["Real", "Fake"]
    )

    ax.set_yticks(
        [0, 1],
        ["Real", "Fake"]
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    fig.colorbar(
        image,
        ax=ax
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrix_test.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"[INFO] Saved: {output_path}"
    )

    return cm


# ============================================================
# 7. ROC CURVE
# ============================================================

def save_roc_curve(
    labels,
    probabilities
):

    fpr, tpr, thresholds = roc_curve(
        labels,
        probabilities
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    print(
        f"\n[INFO] ROC-AUC: "
        f"{roc_auc:.4f}"
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC-AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve - Test Set"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "roc_curve_test.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"[INFO] Saved: {output_path}"
    )


# ============================================================
# 8. METHOD-LEVEL ANALYSIS
# ============================================================

def analyze_methods(
    labels,
    predictions,
    probabilities,
    paths,
    methods
):

    results = []

    dataframe = pd.DataFrame(
        {
            "relative_path": paths,
            "method": methods,
            "label": labels,
            "probability_fake":
                probabilities,
            "prediction":
                predictions
        }
    )

    for method, group in dataframe.groupby(
        "method"
    ):

        group_labels = group[
            "label"
        ].to_numpy()

        group_predictions = group[
            "prediction"
        ].to_numpy()

        group_probabilities = group[
            "probability_fake"
        ].to_numpy()

        sample_count = len(group)

        accuracy = accuracy_score(
            group_labels,
            group_predictions
        )

        avg_probability = (
            group_probabilities.mean()
        )

        predicted_fake_rate = (
            group_predictions.mean()
        )

        # Nếu là Fake method:
        # tỷ lệ dự đoán đúng Fake
        if np.all(group_labels == 1):

            fake_detection_rate = (
                predicted_fake_rate
            )

            false_positive_rate = np.nan

        # Nếu là Real:
        # tỷ lệ bị nhận nhầm Fake
        elif np.all(group_labels == 0):

            fake_detection_rate = np.nan

            false_positive_rate = (
                predicted_fake_rate
            )

        else:

            fake_detection_rate = np.nan

            false_positive_rate = np.nan

        results.append(
            {
                "method": method,
                "samples": sample_count,
                "accuracy": accuracy,
                "avg_probability_fake":
                    avg_probability,
                "predicted_fake_rate":
                    predicted_fake_rate,
                "fake_detection_rate":
                    fake_detection_rate,
                "false_positive_rate":
                    false_positive_rate
            }
        )

    method_df = pd.DataFrame(
        results
    )

    method_df = method_df.sort_values(
        "method"
    )

    output_path = os.path.join(
        RESULTS_DIR,
        "method_results.csv"
    )

    method_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\n========== KẾT QUẢ THEO METHOD =========="
    )

    print(
        method_df.to_string(
            index=False
        )
    )

    print(
        f"\n[INFO] Saved: {output_path}"
    )

    return method_df, dataframe


# ============================================================
# 9. SAVE CLASSIFICATION REPORT
# ============================================================

def save_classification_report(
    labels,
    predictions
):

    report = classification_report(
        labels,
        predictions,
        target_names=[
            "Real",
            "Fake"
        ],
        digits=4,
        zero_division=0
    )

    print(
        "\n========== CLASSIFICATION REPORT =========="
    )

    print(report)

    output_path = os.path.join(
        RESULTS_DIR,
        "classification_report_test.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    print(
        f"[INFO] Saved: {output_path}"
    )


# ============================================================
# 10. SAVE SUMMARY JSON
# ============================================================

def save_summary(
    metrics,
    confusion_matrix_values,
    checkpoint
):

    summary = {
        "checkpoint_epoch":
            checkpoint.get(
                "epoch",
                None
            ),

        "checkpoint_val_auc":
            checkpoint.get(
                "val_auc",
                None
            ),

        "threshold":
            THRESHOLD,

        "test_metrics":
            metrics,

        "confusion_matrix": [
            confusion_matrix_values[0].tolist(),
            confusion_matrix_values[1].tolist()
        ]
    }

    output_path = os.path.join(
        RESULTS_DIR,
        "test_summary.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(
        f"[INFO] Saved: {output_path}"
    )


# ============================================================
# 11. SAVE RAW PREDICTIONS
# ============================================================

def save_predictions(
    labels,
    probabilities,
    predictions,
    paths,
    methods
):

    dataframe = pd.DataFrame(
        {
            "relative_path":
                paths,

            "method":
                methods,

            "true_label":
                labels,

            "probability_fake":
                probabilities,

            "predicted_label":
                predictions
        }
    )

    output_path = os.path.join(
        RESULTS_DIR,
        "test_predictions.csv"
    )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"[INFO] Saved: {output_path}"
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 65)

    print(
        "       DEEPFAKE DETECTION"
    )

    print(
        "       DETAILED TEST EVALUATION"
    )

    print("=" * 65)

    # -----------------------------
    # Kiểm tra thư mục
    # -----------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # -----------------------------
    # Device
    # -----------------------------

    device = get_device()

    # -----------------------------
    # Dataset
    # -----------------------------

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

    # -----------------------------
    # Model
    # -----------------------------

    model, checkpoint = load_model(
        device
    )

    # -----------------------------
    # Evaluation
    # -----------------------------

    (
        labels,
        probabilities,
        predictions,
        paths,
        methods
    ) = evaluate_test(
        model=model,
        dataset=test_dataset,
        loader=test_loader,
        device=device
    )

    # -----------------------------
    # Global metrics
    # -----------------------------

    metrics = calculate_metrics(
        labels,
        predictions,
        probabilities
    )

    print(
        "\n"
    )

    print("=" * 65)

    print(
        "             FINAL TEST METRICS"
    )

    print("=" * 65)

    print(
        f"Accuracy:   "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision:  "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:     "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1-score:   "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC:    "
        f"{metrics['roc_auc']:.4f}"
    )

    # -----------------------------
    # Confusion matrix
    # -----------------------------

    cm = save_confusion_matrix(
        labels,
        predictions
    )

    # -----------------------------
    # ROC curve
    # -----------------------------

    save_roc_curve(
        labels,
        probabilities
    )

    # -----------------------------
    # Method analysis
    # -----------------------------

    analyze_methods(
        labels,
        predictions,
        probabilities,
        paths,
        methods
    )

    # -----------------------------
    # Classification report
    # -----------------------------

    save_classification_report(
        labels,
        predictions
    )

    # -----------------------------
    # Raw predictions
    # -----------------------------

    save_predictions(
        labels,
        probabilities,
        predictions,
        paths,
        methods
    )

    # -----------------------------
    # Summary
    # -----------------------------

    save_summary(
        metrics,
        cm,
        checkpoint
    )

    print("\n")

    print("=" * 65)

    print(
        "       TEST EVALUATION COMPLETED"
    )

    print("=" * 65)


if __name__ == "__main__":
    main()