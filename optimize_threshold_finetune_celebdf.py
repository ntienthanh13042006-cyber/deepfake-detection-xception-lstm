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
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src_finetune.model_finetune import SpatialTemporalXceptionBiLSTM_FineTune
from src_finetune.train_finetune_celebdf import (
    CelebDFNPYDataset,
    VAL_CSV,
    CHECKPOINT_DIR,
    SEQUENCE_LENGTH,
    BATCH_SIZE,
)

RESULTS_DIR = Path(r"D:\Project\results_celebdf_finetune")
PREDICTIONS_FILE = RESULTS_DIR / "finetune_val_predictions.csv"
THRESHOLD_FILE = RESULTS_DIR / "finetune_threshold_analysis.csv"
REPORT_FILE = RESULTS_DIR / "finetune_threshold_analysis_report.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_model():
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy best checkpoint:\n{checkpoint_path}")
    print(f"[INFO] Checkpoint:\n{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = SpatialTemporalXceptionBiLSTM_FineTune(
        sequence_length=SEQUENCE_LENGTH,
        lstm_hidden_size=256,
        lstm_layers=2,
        dropout=0.5,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    epoch = checkpoint.get("epoch", None)
    val_metrics = checkpoint.get("val_metrics", {})
    print(f"[INFO] Best checkpoint epoch: {epoch}")
    if "roc_auc" in val_metrics:
        print(f"[INFO] Best Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
    return model, checkpoint

@torch.no_grad()
def predict_validation(model, loader):
    """Chạy inference trên toàn bộ Validation."""
    all_labels = []
    all_probs = []
    all_paths = []
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(DEVICE, non_blocking=True)
        logits = model(inputs).squeeze(1)
        probabilities = torch.sigmoid(logits)

        all_labels.extend(labels.numpy().astype(int).tolist())
        all_probs.extend(probabilities.cpu().numpy().tolist())

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + len(labels), len(loader.dataset))
        for idx in range(start_idx, end_idx):
            path = loader.dataset.resolved_paths[idx]
            all_paths.append(str(path))

        if (batch_idx + 1) % 100 == 0:
            print(f"[Inference] Batch {batch_idx + 1}/{len(loader)}")

    return np.array(all_labels), np.array(all_probs), all_paths

def calculate_threshold_metrics(y_true, y_prob, threshold):
    """Tính metric cho một threshold. Quy ước: 0 = Real, 1 = Fake"""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision_fake = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_fake = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_real = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_fake = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    specificity = recall_real
    fpr = (fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    youden_j = recall_fake + specificity - 1.0

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision_fake": precision_fake,
        "recall_fake": recall_fake,
        "recall_real": recall_real,
        "f1_fake": f1_fake,
        "balanced_accuracy": balanced_acc,
        "specificity": specificity,
        "fpr": fpr,
        "youden_j": youden_j,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

def main():
    try:
        print("=" * 80)
        print("G7.5 - VALIDATION THRESHOLD OPTIMIZATION")
        print("=" * 80)
        print(f"[INFO] Device: {DEVICE}")
        if torch.cuda.is_available():
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

        print("\n[INFO] Loading Validation...")
        val_dataset = CelebDFNPYDataset(VAL_CSV)
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"[INFO] Validation samples: {len(val_dataset)}")

        model, checkpoint = load_model()

        print("\n[INFO] Running Validation inference...")
        y_true, y_prob, paths = predict_validation(model, val_loader)

        if len(y_true) != len(val_dataset):
            raise RuntimeError("Số lượng prediction không khớp Validation.")
        if len(paths) != len(val_dataset):
            raise RuntimeError("Số lượng paths không khớp Validation.")

        val_auc = roc_auc_score(y_true, y_prob)
        print(f"\n[INFO] Validation ROC-AUC: {val_auc:.4f}")

        prediction_df = pd.DataFrame({
            "npy_path": paths,
            "label": y_true,
            "fake_probability": y_prob,
        })
        prediction_df.to_csv(PREDICTIONS_FILE, index=False)
        print(f"[SAVE] Validation predictions:\n{PREDICTIONS_FILE}")

        thresholds = np.arange(0.01, 1.00, 0.01)
        results = []
        for threshold in thresholds:
            metrics = calculate_threshold_metrics(y_true, y_prob, float(threshold))
            results.append(metrics)

        results_df = pd.DataFrame(results)
        results_df.to_csv(THRESHOLD_FILE, index=False)

        best_balanced_idx = results_df["balanced_accuracy"].idxmax()
        best_balanced = results_df.loc[best_balanced_idx]

        best_youden_idx = results_df["youden_j"].idxmax()
        best_youden = results_df.loc[best_youden_idx]

        best_f1_idx = results_df["f1_fake"].idxmax()
        best_f1 = results_df.loc[best_f1_idx]

        target_recalls = [0.70, 0.75, 0.80]
        target_results = []
        for target in target_recalls:
            candidates = results_df[results_df["recall_real"] >= target]
            if len(candidates) == 0:
                target_results.append({
                    "target_real_recall": target,
                    "threshold": np.nan,
                    "real_recall": np.nan,
                    "fake_recall": np.nan,
                    "balanced_accuracy": np.nan,
                    "f1_fake": np.nan,
                })
                continue
            
            best_idx = candidates["balanced_accuracy"].idxmax()
            row = candidates.loc[best_idx]
            target_results.append({
                "target_real_recall": target,
                "threshold": row["threshold"],
                "real_recall": row["recall_real"],
                "fake_recall": row["recall_fake"],
                "balanced_accuracy": row["balanced_accuracy"],
                "f1_fake": row["f1_fake"],
            })

        target_df = pd.DataFrame(target_results)

        with open(REPORT_FILE, "w", encoding="utf-8") as report:
            report.write("G7.5 - VALIDATION THRESHOLD OPTIMIZATION\n")
            report.write("=" * 80 + "\n")
            report.write(f"Checkpoint epoch: {checkpoint.get('epoch')}\n")
            report.write(f"Validation ROC-AUC: {val_auc:.6f}\n")
            report.write(f"Validation samples: {len(y_true)}\n\n")

            report.write("\nBEST BALANCED ACCURACY\n")
            report.write("-" * 80 + "\n")
            report.write(best_balanced.to_string())
            report.write("\n")

            report.write("\nBEST YOUDEN J\n")
            report.write("-" * 80 + "\n")
            report.write(best_youden.to_string())
            report.write("\n")

            report.write("\nBEST F1\n")
            report.write("-" * 80 + "\n")
            report.write(best_f1.to_string())
            report.write("\n")

            report.write("\nTARGET REAL RECALL\n")
            report.write("-" * 80 + "\n")
            report.write(target_df.to_string(index=False))
            report.write("\n")

        print("\n")
        print("=" * 80)
        print("BEST BALANCED ACCURACY")
        print("=" * 80)
        print(f"Threshold         : {best_balanced['threshold']:.2f}")
        print(f"Accuracy          : {best_balanced['accuracy']:.4f}")
        print(f"Real Recall       : {best_balanced['recall_real']:.4f}")
        print(f"Fake Recall       : {best_balanced['recall_fake']:.4f}")
        print(f"F1                : {best_balanced['f1_fake']:.4f}")
        print(f"Balanced Accuracy : {best_balanced['balanced_accuracy']:.4f}")
        print(f"Youden J          : {best_balanced['youden_j']:.4f}")
        print(f"CM                : TN={int(best_balanced['tn'])}, FP={int(best_balanced['fp'])}, FN={int(best_balanced['fn'])}, TP={int(best_balanced['tp'])}")

        print("\n")
        print("=" * 80)
        print("BEST YOUDEN J")
        print("=" * 80)
        print(f"Threshold         : {best_youden['threshold']:.2f}")
        print(f"Real Recall       : {best_youden['recall_real']:.4f}")
        print(f"Fake Recall       : {best_youden['recall_fake']:.4f}")
        print(f"Balanced Accuracy : {best_youden['balanced_accuracy']:.4f}")

        print("\n")
        print("=" * 80)
        print("BEST F1")
        print("=" * 80)
        print(f"Threshold         : {best_f1['threshold']:.2f}")
        print(f"Real Recall       : {best_f1['recall_real']:.4f}")
        print(f"Fake Recall       : {best_f1['recall_fake']:.4f}")
        print(f"F1                : {best_f1['f1_fake']:.4f}")

        print("\n")
        print("=" * 80)
        print("TARGET REAL RECALL")
        print("=" * 80)
        print(target_df.to_string(index=False))

        print("\n")
        print("=" * 80)
        print("G7.5 HOÀN TẤT")
        print("=" * 80)
        print(f"[SAVE] {PREDICTIONS_FILE}")
        print(f"[SAVE] {THRESHOLD_FILE}")
        print(f"[SAVE] {REPORT_FILE}")
        print("\n[NEXT] Sau khi xác nhận threshold, mới chuyển sang G7.6 Test 518.")

    except Exception as e:
        print("\n[ERROR] Threshold optimization thất bại.")
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()