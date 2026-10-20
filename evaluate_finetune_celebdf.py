"""
============================================================
G7.6 - ĐÁNH GIÁ FINE-TUNED MODEL TRÊN TEST 518
CELEB-DF V2
============================================================

Mục tiêu:
- Load best checkpoint của Experiment B.
- Sử dụng threshold ĐÃ CHỐT từ Validation.
- Đánh giá đúng 518 video Test chính thức.
- KHÔNG tối ưu threshold trên Test.

Checkpoint:
    D:\\Project\\checkpoints_celebdf_finetune\\best_model.pth

Threshold:
    0.98

Label:
    0 = Real
    1 = Fake

Metrics:
    Accuracy
    Precision
    Recall Real
    Recall Fake
    F1
    Balanced Accuracy
    ROC-AUC
    Confusion Matrix

Phân tích thêm:
    Celeb-real
    YouTube-real
    Celeb-synthesis
============================================================
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from torch.utils.data import DataLoader

from model_finetune import (
    SpatialTemporalXceptionBiLSTM_FineTune
)

from train_finetune_celebdf import (
    CelebDFNPYDataset,
    PROCESSED_ROOT,
    CHECKPOINT_DIR,
    SEQUENCE_LENGTH,
)


# ============================================================
# 1. CẤU HÌNH
# ============================================================

PROJECT_DIR = Path(
    r"D:\Project"
)

TEST_CSV = (
    PROJECT_DIR
    / "results_celebdf"
    / "processed_test.csv"
)

# ------------------------------------------------------------
# Threshold đã được chọn từ Validation
# ------------------------------------------------------------

THRESHOLD = 0.98

# ------------------------------------------------------------
# Batch size
# ------------------------------------------------------------

BATCH_SIZE = 2

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

RESULTS_DIR = (
    PROJECT_DIR
    / "results_celebdf_finetune"
)

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "finetune_test_predictions.csv"
)

REPORT_FILE = (
    RESULTS_DIR
    / "finetune_test_evaluation_report.txt"
)

CONFUSION_MATRIX_FILE = (
    RESULTS_DIR
    / "finetune_test_confusion_matrix.csv"
)


# ============================================================
# 2. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 3. TẠO THƯ MỤC OUTPUT
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. LOAD MODEL
# ============================================================

def load_model():
    """
    Load best checkpoint của Experiment B.
    """

    checkpoint_path = (
        CHECKPOINT_DIR
        / "best_model.pth"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Không tìm thấy checkpoint:\n"
            f"{checkpoint_path}"
        )

    print(
        "[INFO] Checkpoint:"
    )

    print(
        checkpoint_path
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model = (
        SpatialTemporalXceptionBiLSTM_FineTune(
            sequence_length=SEQUENCE_LENGTH,
            lstm_hidden_size=256,
            lstm_layers=2,
            dropout=0.5,
        )
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    checkpoint_epoch = (
        checkpoint.get(
            "epoch",
            None
        )
    )

    checkpoint_metrics = (
        checkpoint.get(
            "val_metrics",
            {}
        )
    )

    print(
        f"[INFO] Best checkpoint epoch: "
        f"{checkpoint_epoch}"
    )

    if (
        "roc_auc"
        in checkpoint_metrics
    ):

        print(
            "[INFO] Checkpoint "
            "Validation ROC-AUC: "
            f"{checkpoint_metrics['roc_auc']:.4f}"
        )

    return (
        model,
        checkpoint
    )


# ============================================================
# 5. XÁC ĐỊNH SOURCE
# ============================================================

def extract_source(npy_path: str) -> str:
    """
    Xác định nguồn dữ liệu từ đường dẫn NPY.

    Dự kiến:
        Real\\Celeb-real\\...
        Real\\YouTube-real\\...
        Fake\\Celeb-synthesis\\...
    """

    normalized = (
        str(npy_path)
        .replace("\\", "/")
    )

    if (
        "/Celeb-real/"
        in normalized
    ):

        return "Celeb-real"

    if (
        "/YouTube-real/"
        in normalized
    ):

        return "YouTube-real"

    if (
        "/Celeb-synthesis/"
        in normalized
    ):

        return "Celeb-synthesis"

    return "Unknown"


# ============================================================
# 6. INFERENCE TEST
# ============================================================

@torch.no_grad()
def run_inference(
    model,
    dataset,
    loader,
):
    """
    Chạy inference toàn bộ Test 518.

    Không thay đổi threshold.
    """

    all_labels = []

    all_probabilities = []

    all_paths = []

    for batch_idx, (
        inputs,
        labels
    ) in enumerate(loader):

        inputs = inputs.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits = model(
            inputs
        )

        logits = logits.squeeze(
            1
        )

        probabilities = torch.sigmoid(
            logits
        )

        # ----------------------------------------------------
        # Lưu labels
        # ----------------------------------------------------

        all_labels.extend(
            labels.numpy()
            .astype(int)
            .tolist()
        )

        # ----------------------------------------------------
        # Lưu probability
        # ----------------------------------------------------

        all_probabilities.extend(
            probabilities.cpu()
            .numpy()
            .tolist()
        )

        # ----------------------------------------------------
        # Lưu paths
        # ----------------------------------------------------

        start_idx = (
            batch_idx
            * BATCH_SIZE
        )

        end_idx = min(
            start_idx
            + len(labels),
            len(dataset)
        )

        for idx in range(
            start_idx,
            end_idx
        ):

            all_paths.append(
                str(
                    dataset
                    .resolved_paths[idx]
                )
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (batch_idx + 1) % 50
            == 0
        ):

            print(
                f"[Inference] "
                f"Batch "
                f"{batch_idx + 1}/"
                f"{len(loader)}"
            )

    return (
        np.array(
            all_labels
        ),
        np.array(
            all_probabilities
        ),
        all_paths,
    )


# ============================================================
# 7. TÍNH METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_probability,
    threshold,
):
    """
    Tính toàn bộ metrics ở threshold cố định.
    """

    y_pred = (
        y_probability
        >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = (
        accuracy_score(
            y_true,
            y_pred
        )
    )

    # --------------------------------------------------------
    # Fake = Positive class
    # --------------------------------------------------------

    precision_fake = (
        precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0
        )
    )

    recall_fake = (
        recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0
        )
    )

    f1_fake = (
        f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Real recall
    # --------------------------------------------------------

    recall_real = (
        recall_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Real precision
    # --------------------------------------------------------

    precision_real = (
        precision_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Balanced Accuracy
    # --------------------------------------------------------

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred
        )
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    roc_auc = (
        roc_auc_score(
            y_true,
            y_probability
        )
    )

    # --------------------------------------------------------
    # FPR / Specificity
    # --------------------------------------------------------

    if (
        tn + fp
    ) > 0:

        specificity = (
            tn / (tn + fp)
        )

        fpr = (
            fp / (tn + fp)
        )

    else:

        specificity = 0.0
        fpr = 0.0

    return {
        "threshold":
            threshold,

        "accuracy":
            accuracy,

        "precision_fake":
            precision_fake,

        "recall_fake":
            recall_fake,

        "f1_fake":
            f1_fake,

        "precision_real":
            precision_real,

        "recall_real":
            recall_real,

        "specificity":
            specificity,

        "fpr":
            fpr,

        "balanced_accuracy":
            balanced_accuracy,

        "roc_auc":
            roc_auc,

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),
    }


# ============================================================
# 8. PHÂN TÍCH THEO SOURCE
# ============================================================

def calculate_source_metrics(
    prediction_df
):
    """
    Tính accuracy/recall theo từng source.
    """

    source_results = []

    for source, group in (
        prediction_df
        .groupby("source")
    ):

        y_true = (
            group["label"]
            .astype(int)
            .values
        )

        y_pred = (
            group["prediction"]
            .astype(int)
            .values
        )

        y_prob = (
            group["fake_probability"]
            .astype(float)
            .values
        )

        total = len(group)

        correct = int(
            np.sum(
                y_true == y_pred
            )
        )

        accuracy = (
            correct / total
            if total > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Real source
        # ----------------------------------------------------

        if set(y_true) == {0}:

            real_recall = (
                recall_score(
                    y_true,
                    y_pred,
                    pos_label=0,
                    zero_division=0
                )
            )

            fake_recall = np.nan

        # ----------------------------------------------------
        # Fake source
        # ----------------------------------------------------

        elif set(y_true) == {1}:

            fake_recall = (
                recall_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0
                )
            )

            real_recall = np.nan

        else:

            real_recall = (
                recall_score(
                    y_true,
                    y_pred,
                    pos_label=0,
                    zero_division=0
                )
            )

            fake_recall = (
                recall_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0
                )
            )

        source_results.append(
            {
                "source":
                    source,

                "total":
                    total,

                "correct":
                    correct,

                "incorrect":
                    total - correct,

                "accuracy":
                    accuracy,

                "real_recall":
                    real_recall,

                "fake_recall":
                    fake_recall,

                "mean_fake_probability":
                    float(
                        np.mean(
                            y_prob
                        )
                    ),
            }
        )

    return pd.DataFrame(
        source_results
    )


# ============================================================
# 9. MAIN
# ============================================================

def main():

    try:

        print("=" * 80)
        print(
            "G7.6 - TEST EVALUATION "
            "FINE-TUNED XCEPTION + BI-LSTM"
        )
        print("=" * 80)

        # ====================================================
        # Device
        # ====================================================

        print(
            f"[INFO] Device: "
            f"{DEVICE}"
        )

        if torch.cuda.is_available():

            print(
                "[INFO] GPU: "
                f"{torch.cuda.get_device_name(0)}"
            )

            print(
                "[INFO] CUDA: "
                f"{torch.version.cuda}"
            )

        # ====================================================
        # Threshold
        # ====================================================

        print("\n" + "=" * 80)
        print(
            "THRESHOLD ĐÃ CHỐT TỪ VALIDATION"
        )
        print("=" * 80)

        print(
            f"Threshold = {THRESHOLD:.2f}"
        )

        print(
            "[IMPORTANT] "
            "Không tối ưu threshold trên Test."
        )

        print("=" * 80)

        # ====================================================
        # Test CSV
        # ====================================================

        if not TEST_CSV.exists():

            raise FileNotFoundError(
                "Không tìm thấy Test CSV:\n"
                f"{TEST_CSV}"
            )

        print(
            f"\n[INFO] Test CSV:\n"
            f"{TEST_CSV}"
        )

        # ====================================================
        # Dataset
        # ====================================================

        print(
            "\n[INFO] Loading Test..."
        )

        test_dataset = (
            CelebDFNPYDataset(
                TEST_CSV
            )
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=(
                torch.cuda.is_available()
            ),
        )

        print(
            f"[INFO] Test samples: "
            f"{len(test_dataset)}"
        )

        # ====================================================
        # Kiểm tra đúng 518
        # ====================================================

        if len(test_dataset) != 518:

            raise ValueError(
                "Test set phải có đúng "
                f"518 video, nhưng nhận "
                f"{len(test_dataset)}."
            )

        # ====================================================
        # Class distribution
        # ====================================================

        test_labels = (
            test_dataset.df[
                test_dataset.label_column
            ]
            .astype(int)
            .values
        )

        real_count = int(
            np.sum(
                test_labels == 0
            )
        )

        fake_count = int(
            np.sum(
                test_labels == 1
            )
        )

        print("\n" + "=" * 80)
        print(
            "TEST CLASS DISTRIBUTION"
        )
        print("=" * 80)

        print(
            f"Real: {real_count}"
        )

        print(
            f"Fake: {fake_count}"
        )

        print("=" * 80)

        # ====================================================
        # Load model
        # ====================================================

        model, checkpoint = (
            load_model()
        )

        # ====================================================
        # Inference
        # ====================================================

        print("\n[INFO] Running Test inference...")

        (
            y_true,
            y_probability,
            paths,
        ) = run_inference(
            model,
            test_dataset,
            test_loader
        )

        # ====================================================
        # Sanity check
        # ====================================================

        if len(y_true) != 518:

            raise RuntimeError(
                "Số prediction không bằng "
                "518."
            )

        if len(paths) != 518:

            raise RuntimeError(
                "Số path prediction "
                "không bằng 518."
            )

        # ====================================================
        # Prediction bằng threshold 0.98
        # ====================================================

        y_prediction = (
            y_probability
            >= THRESHOLD
        ).astype(int)

        # ====================================================
        # Overall metrics
        # ====================================================

        metrics = calculate_metrics(
            y_true,
            y_probability,
            THRESHOLD
        )

        # ====================================================
        # Prediction dataframe
        # ====================================================

        prediction_df = pd.DataFrame({

            "npy_path":
                paths,

            "label":
                y_true,

            "fake_probability":
                y_probability,

            "prediction":
                y_prediction,

        })

        # ----------------------------------------------------
        # Source
        # ----------------------------------------------------

        prediction_df[
            "source"
        ] = prediction_df[
            "npy_path"
        ].apply(
            extract_source
        )

        # ----------------------------------------------------
        # Correct / wrong
        # ----------------------------------------------------

        prediction_df[
            "correct"
        ] = (
            prediction_df[
                "label"
            ]
            ==
            prediction_df[
                "prediction"
            ]
        )

        # ----------------------------------------------------
        # Error type
        # ----------------------------------------------------

        def get_error_type(row):

            if row["correct"]:

                return "Correct"

            if (
                row["label"] == 0
                and row["prediction"] == 1
            ):

                return "Real_to_Fake"

            if (
                row["label"] == 1
                and row["prediction"] == 0
            ):

                return "Fake_to_Real"

            return "Unknown"

        prediction_df[
            "error_type"
        ] = prediction_df.apply(
            get_error_type,
            axis=1
        )

        # ====================================================
        # Save predictions
        # ====================================================

        prediction_df.to_csv(
            PREDICTIONS_FILE,
            index=False
        )

        print(
            "\n[SAVE] Test predictions:\n"
            f"{PREDICTIONS_FILE}"
        )

        # ====================================================
        # Source metrics
        # ====================================================

        source_df = (
            calculate_source_metrics(
                prediction_df
            )
        )

        # ====================================================
        # Save confusion matrix
        # ====================================================

        cm = confusion_matrix(
            y_true,
            y_prediction,
            labels=[0, 1]
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual_Real",
                "Actual_Fake",
            ],
            columns=[
                "Predicted_Real",
                "Predicted_Fake",
            ]
        )

        cm_df.to_csv(
            CONFUSION_MATRIX_FILE
        )

        # ====================================================
        # Report
        # ====================================================

        with open(
            REPORT_FILE,
            "w",
            encoding="utf-8"
        ) as report:

            report.write(
                "G7.6 - TEST EVALUATION\n"
            )

            report.write(
                "=" * 80
                + "\n"
            )

            report.write(
                f"Checkpoint epoch: "
                f"{checkpoint.get('epoch')}\n"
            )

            report.write(
                f"Threshold: "
                f"{THRESHOLD:.2f}\n"
            )

            report.write(
                f"Test samples: "
                f"{len(y_true)}\n"
            )

            report.write(
                f"Real: {real_count}\n"
            )

            report.write(
                f"Fake: {fake_count}\n"
            )

            # ------------------------------------------------
            # Overall
            # ------------------------------------------------

            report.write(
                "\n"
                "OVERALL METRICS\n"
            )

            report.write(
                "-" * 80
                + "\n"
            )

            for key, value in (
                metrics.items()
            ):

                report.write(
                    f"{key}: "
                    f"{value}\n"
                )

            # ------------------------------------------------
            # Confusion matrix
            # ------------------------------------------------

            report.write(
                "\n"
                "CONFUSION MATRIX\n"
            )

            report.write(
                "-" * 80
                + "\n"
            )

            report.write(
                cm_df.to_string()
            )

            # ------------------------------------------------
            # Source metrics
            # ------------------------------------------------

            report.write(
                "\n\n"
                "SOURCE-LEVEL METRICS\n"
            )

            report.write(
                "-" * 80
                + "\n"
            )

            report.write(
                source_df.to_string(
                    index=False
                )
            )

            # ------------------------------------------------
            # Error counts
            # ------------------------------------------------

            real_to_fake = int(
                np.sum(
                    (
                        y_true == 0
                    )
                    &
                    (
                        y_prediction == 1
                    )
                )
            )

            fake_to_real = int(
                np.sum(
                    (
                        y_true == 1
                    )
                    &
                    (
                        y_prediction == 0
                    )
                )
            )

            correct_count = int(
                np.sum(
                    y_true
                    ==
                    y_prediction
                )
            )

            wrong_count = (
                len(y_true)
                -
                correct_count
            )

            report.write(
                "\n\nERROR ANALYSIS\n"
            )

            report.write(
                "-" * 80
                + "\n"
            )

            report.write(
                f"Correct        : "
                f"{correct_count}\n"
            )

            report.write(
                f"Wrong          : "
                f"{wrong_count}\n"
            )

            report.write(
                f"Real -> Fake   : "
                f"{real_to_fake}\n"
            )

            report.write(
                f"Fake -> Real   : "
                f"{fake_to_real}\n"
            )

        # ====================================================
        # Console output
        # ====================================================

        print("\n")
        print("=" * 80)
        print(
            "G7.6 - KẾT QUẢ TEST"
        )
        print("=" * 80)

        print(
            f"Threshold           : "
            f"{THRESHOLD:.2f}"
        )

        print(
            f"Accuracy            : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision Fake      : "
            f"{metrics['precision_fake']:.4f}"
        )

        print(
            f"Recall Real         : "
            f"{metrics['recall_real']:.4f}"
        )

        print(
            f"Recall Fake         : "
            f"{metrics['recall_fake']:.4f}"
        )

        print(
            f"F1 Fake             : "
            f"{metrics['f1_fake']:.4f}"
        )

        print(
            f"Balanced Accuracy   : "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"ROC-AUC             : "
            f"{metrics['roc_auc']:.4f}"
        )

        print(
            f"Specificity         : "
            f"{metrics['specificity']:.4f}"
        )

        print(
            f"FPR                 : "
            f"{metrics['fpr']:.4f}"
        )

        print("\nConfusion Matrix:")

        print(
            cm_df
        )

        print("\n")
        print(
            "=" * 80
        )

        print(
            "SOURCE-LEVEL METRICS"
        )

        print(
            "=" * 80
        )

        print(
            source_df.to_string(
                index=False
            )
        )

        print("\n")
        print(
            "=" * 80
        )

        print(
            "G7.6 HOÀN TẤT"
        )

        print(
            "=" * 80
        )

        print(
            "[SAVE] "
            f"{PREDICTIONS_FILE}"
        )

        print(
            "[SAVE] "
            f"{REPORT_FILE}"
        )

        print(
            "[SAVE] "
            f"{CONFUSION_MATRIX_FILE}"
        )

        print("\n[NEXT] G7.7:")
        print(
            "So sánh Baseline vs Fine-tuning."
        )

    except KeyboardInterrupt:

        print(
            "\n[INFO] "
            "Người dùng dừng inference."
        )

        sys.exit(1)

    except Exception as e:

        print(
            "\n[ERROR] "
            "Test evaluation thất bại."
        )

        print(
            f"[ERROR] {e}"
        )

        import traceback

        traceback.print_exc()

        sys.exit(1)


if __name__ == "__main__":

    main()