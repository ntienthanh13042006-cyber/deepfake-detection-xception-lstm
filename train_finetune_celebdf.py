"""
============================================================
G7.4 - TRAIN FINE-TUNING XCEPTION + BI-LSTM
TRÊN CELEB-DF V2
============================================================

Mục tiêu:
- Giữ nguyên kiến trúc Xception + Bi-LSTM baseline.
- Chỉ fine-tune phần cuối của Xception.
- So sánh công bằng với baseline trước đó.

Chiến lược fine-tuning:
    Xception:
        block1  -> block11 : FROZEN
        block12            : TRAINABLE
        conv3              : TRAINABLE
        bn3                : TRAINABLE affine parameters
        conv4              : TRAINABLE
        bn4                : TRAINABLE affine parameters

Learning rate:
    Xception fine-tuning      = 1e-5
    Bi-LSTM + Classifier      = 1e-4

Dataset:
    Train = 4808
    Val   = 1203
    Test  = 518 (KHÔNG dùng trong training)

Input:
    (B, 15, 3, 224, 224)

Output:
    (B, 1) - Logit

Label:
    0 = Real
    1 = Fake
============================================================
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
)

# Model fine-tuning
from src_finetune.model_finetune import SpatialTemporalXceptionBiLSTM_FineTune

# ============================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ============================================================
PROJECT_DIR = Path(r"D:\Project")

PROCESSED_ROOT = PROJECT_DIR / "Processed_Data_CelebDF"
TRAIN_CSV = PROJECT_DIR / "results_celebdf" / "processed_train.csv"
VAL_CSV = PROJECT_DIR / "results_celebdf" / "processed_val.csv"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints_celebdf_finetune"

# ============================================================
# 2. CẤU HÌNH DỮ LIỆU
# ============================================================
SEQUENCE_LENGTH = 15
IMAGE_SIZE = 224

# ============================================================
# 3. CẤU HÌNH TRAINING
# ============================================================
BATCH_SIZE = 2
NUM_WORKERS = 0
MAX_EPOCHS = 15

XCEPTION_LR = 1e-5
LSTM_CLASSIFIER_LR = 1e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 4
SEED = 42

# ============================================================
# 4. RANDOM SEED
# ============================================================
def set_seed(seed: int = 42):
    """Thiết lập random seed để tăng tính tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# 5. RESOLVE ĐƯỜNG DẪN NPY
# ============================================================
def resolve_npy_path(raw_path: str) -> Path:
    """Chuyển đường dẫn trong CSV thành đường dẫn thực tế."""
    raw_path = str(raw_path).strip()
    if not raw_path:
        raise ValueError("Đường dẫn NPY trong CSV đang rỗng.")
    path_obj = Path(raw_path)
    if path_obj.is_absolute():
        return path_obj
    return PROCESSED_ROOT / path_obj

# ============================================================
# 6. DATASET
# ============================================================
class CelebDFNPYDataset(Dataset):
    """
    Dataset đọc NPY đã preprocessing.
    Input NPY: (15, 224, 224, 3), uint8
    Output: (15, 3, 224, 224), float32
    Label: 0 = Real, 1 = Fake
    """
    def __init__(self, csv_path: Path):
        if not csv_path.exists():
            raise FileNotFoundError(f"Không tìm thấy CSV:\n{csv_path}")

        if not PROCESSED_ROOT.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục NPY:\n{PROCESSED_ROOT}")

        self.df = pd.read_csv(csv_path)
        if len(self.df) == 0:
            raise ValueError(f"CSV không có dữ liệu:\n{csv_path}")

        possible_path_columns = ["npy_path", "path", "processed_path", "file_path"]
        self.path_column = None
        for column in possible_path_columns:
            if column in self.df.columns:
                self.path_column = column
                break
        if self.path_column is None:
            raise ValueError(f"Không tìm thấy cột đường dẫn NPY. Các cột hiện có: {list(self.df.columns)}")

        possible_label_columns = ["label", "target", "class"]
        self.label_column = None
        for column in possible_label_columns:
            if column in self.df.columns:
                self.label_column = column
                break
        if self.label_column is None:
            raise ValueError(f"Không tìm thấy cột label. Các cột hiện có: {list(self.df.columns)}")

        self.df[self.label_column] = self.df[self.label_column].astype(int)
        unique_labels = set(self.df[self.label_column].unique().tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Label không hợp lệ: {unique_labels}\nQuy ước: 0=Real, 1=Fake.")

        print("\n" + "=" * 80)
        print(f"KIỂM TRA ĐƯỜNG DẪN NPY - {csv_path.name}")
        print("=" * 80)

        self.resolved_paths = []
        missing_files = []
        for index, raw_path in enumerate(self.df[self.path_column].astype(str)):
            try:
                resolved_path = resolve_npy_path(raw_path)
            except Exception as e:
                missing_files.append((index, raw_path, f"Resolve error: {e}"))
                continue

            self.resolved_paths.append(resolved_path)
            if not resolved_path.exists():
                missing_files.append((index, raw_path, str(resolved_path)))

        if len(self.resolved_paths) != len(self.df):
            raise RuntimeError("Số lượng đường dẫn resolve được không khớp số lượng dòng trong CSV.")

        total_files = len(self.resolved_paths)
        missing_count = len(missing_files)
        existing_count = total_files - missing_count

        print(f"[INFO] Tổng NPY : {total_files}")
        print(f"[INFO] Tồn tại  : {existing_count}")
        print(f"[INFO] Thiếu    : {missing_count}")

        print("\n[INFO] Một số path sau khi resolve:")
        for i in range(min(3, len(self.resolved_paths))):
            print(f"  [{i}] {self.resolved_paths[i]}")

        if missing_count > 0:
            print("\n" + "=" * 80)
            print("DANH SÁCH FILE NPY BỊ THIẾU")
            print("=" * 80)
            for item in missing_files[:20]:
                index, raw_path, resolved = item
                print(f"\n[MISSING] index={index}")
                print(f"  CSV     : {raw_path}")
                print(f"  Resolved: {resolved}")
            if missing_count > 20:
                print(f"\n... còn {missing_count - 20} file khác.")
            raise FileNotFoundError(f"Có {missing_count}/{total_files} file NPY không tồn tại.")

        print(f"[PASS] Tất cả {total_files} đường dẫn NPY hợp lệ.")
        print("=" * 80)

        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        npy_path = self.resolved_paths[idx]
        row = self.df.iloc[idx]
        label = int(row[self.label_column])

        try:
            frames = np.load(npy_path)
        except Exception as e:
            raise RuntimeError(f"Không thể đọc NPY:\n{npy_path}\nLỗi: {e}")

        expected_shape = (SEQUENCE_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 3)
        if frames.shape != expected_shape:
            raise ValueError(f"Shape NPY không đúng.\nFile      : {npy_path}\nNhận      : {frames.shape}\nKỳ vọng   : {expected_shape}")

        frames = frames.astype(np.float32) / 255.0
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        frames = (frames - self.mean) / self.std
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return frames, label_tensor

# ============================================================
# 7. METRICS
# ============================================================
def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray):
    """Tính metric tại threshold 0.5."""
    predictions = (probabilities >= 0.5).astype(int)
    labels = labels.astype(int)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    try:
        auc = roc_auc_score(labels, probabilities)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
    }

# ============================================================
# 8. TRAIN ONE EPOCH
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Huấn luyện một epoch."""
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(inputs).squeeze(1)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        current_batch_size = inputs.size(0)
        total_loss += loss.item() * current_batch_size
        probabilities = torch.sigmoid(logits)

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_probs.extend(probabilities.detach().cpu().numpy().tolist())

        if (batch_idx + 1) % 500 == 0:
            print(f"    [Train] Batch {batch_idx + 1}/{len(loader)} | Loss = {loss.item():.4f}")

    epoch_loss = total_loss / len(loader.dataset)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_probs))
    metrics["loss"] = epoch_loss
    return metrics

# ============================================================
# 9. VALIDATION
# ============================================================
@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validation toàn bộ validation set."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(inputs).squeeze(1)
            loss = criterion(logits, labels)

        current_batch_size = inputs.size(0)
        total_loss += loss.item() * current_batch_size
        probabilities = torch.sigmoid(logits)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probabilities.cpu().numpy().tolist())

    epoch_loss = total_loss / len(loader.dataset)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_probs))
    metrics["loss"] = epoch_loss
    return metrics

# ============================================================
# 10. BUILD OPTIMIZER
# ============================================================
def build_optimizer(model):
    """Differential learning rate."""
    xception_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("xception."):
            xception_params.append(param)
        else:
            other_params.append(param)

    if len(xception_params) == 0:
        raise RuntimeError("Không có parameter Xception nào được fine-tune.")
    if len(other_params) == 0:
        raise RuntimeError("Không có parameter Bi-LSTM/classifier để train.")

    optimizer = torch.optim.Adam(
        [
            {"params": xception_params, "lr": XCEPTION_LR},
            {"params": other_params, "lr": LSTM_CLASSIFIER_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    print("\n" + "=" * 80)
    print("OPTIMIZER")
    print("=" * 80)
    print(f"Xception trainable params : {sum(p.numel() for p in xception_params):,}")
    print(f"LSTM + classifier params  : {sum(p.numel() for p in other_params):,}")
    print(f"Xception LR               : {XCEPTION_LR}")
    print(f"LSTM/classifier LR        : {LSTM_CLASSIFIER_LR}")
    print(f"Weight decay              : {WEIGHT_DECAY}")
    print("=" * 80)

    return optimizer

# ============================================================
# 11. SAVE CHECKPOINT
# ============================================================
def save_checkpoint(path, model, optimizer, epoch, val_metrics):
    """Lưu best checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": val_metrics,
        "config": {
            "sequence_length": SEQUENCE_LENGTH,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "xception_lr": XCEPTION_LR,
            "lstm_classifier_lr": LSTM_CLASSIFIER_LR,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "seed": SEED,
            "fine_tuned": [
                "xception.block12",
                "xception.conv3",
                "xception.bn3",
                "xception.conv4",
                "xception.bn4",
            ],
            "frozen": "xception.block1 -> block11",
            "processed_root": str(PROCESSED_ROOT),
        },
    }
    torch.save(checkpoint, path)

# ============================================================
# 12. MAIN
# ============================================================
def main():
    try:
        print("=" * 80)
        print("G7.4 - TRAIN FINE-TUNING XCEPTION + BI-LSTM")
        print("=" * 80)

        set_seed(SEED)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Device: {device}")
        if torch.cuda.is_available():
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] CUDA: {torch.version.cuda}")

        print("\n" + "=" * 80)
        print("KIỂM TRA ĐƯỜNG DẪN CHÍNH")
        print("=" * 80)
        print(f"[INFO] Project      : {PROJECT_DIR}")
        print(f"[INFO] Processed    : {PROCESSED_ROOT}")
        print(f"[INFO] Train CSV    : {TRAIN_CSV}")
        print(f"[INFO] Val CSV      : {VAL_CSV}")
        print(f"[INFO] Checkpoint   : {CHECKPOINT_DIR}")

        if not PROCESSED_ROOT.exists():
            raise FileNotFoundError(f"Thư mục Processed_Data_CelebDF không tồn tại:\n{PROCESSED_ROOT}")

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        print("\n[INFO] Loading dataset...")
        train_dataset = CelebDFNPYDataset(TRAIN_CSV)
        val_dataset = CelebDFNPYDataset(VAL_CSV)

        print(f"\n[INFO] Train samples: {len(train_dataset)}")
        print(f"[INFO] Val samples  : {len(val_dataset)}")

        train_labels = train_dataset.df[train_dataset.label_column].astype(int).values
        real_count = int(np.sum(train_labels == 0))
        fake_count = int(np.sum(train_labels == 1))

        print("\n" + "=" * 80)
        print("TRAIN CLASS DISTRIBUTION")
        print("=" * 80)
        print(f"Real: {real_count}")
        print(f"Fake: {fake_count}")
        print("=" * 80)

        class_counts = np.bincount(train_labels, minlength=2)
        if np.any(class_counts == 0):
            raise ValueError("Train dataset phải có cả Real và Fake.")

        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels]
        sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        print("\n" + "=" * 80)
        print("DATALOADER")
        print("=" * 80)
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches  : {len(val_loader)}")
        print(f"Batch size   : {BATCH_SIZE}")
        print(f"Workers      : {NUM_WORKERS}")
        print("=" * 80)

        print("\n[INFO] Khởi tạo model Fine-tuning...")
        model = SpatialTemporalXceptionBiLSTM_FineTune(
            sequence_length=SEQUENCE_LENGTH,
            lstm_hidden_size=256,
            lstm_layers=2,
            dropout=0.5,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = build_optimizer(model)
        scaler = GradScaler(enabled=(device.type == "cuda"))

        best_val_auc = -1.0
        best_epoch = -1
        patience_counter = 0

        history = []

        print("\n" + "=" * 80)
        print("BẮT ĐẦU TRAINING FINE-TUNING")
        print("=" * 80)

        start_time = time.time()

        for epoch in range(1, MAX_EPOCHS + 1):
            epoch_start = time.time()
            print(f"\n--- EPOCH {epoch}/{MAX_EPOCHS} ---")

            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
            val_metrics = validate(model, val_loader, criterion, device)

            epoch_time = time.time() - epoch_start

            print(f"[Train] Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | ROC-AUC: {train_metrics['roc_auc']:.4f}")
            print(f"[Val]   Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | ROC-AUC: {val_metrics['roc_auc']:.4f}")
            print(f"Thời gian epoch: {epoch_time:.2f}s")

            epoch_info = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "train_auc": train_metrics["roc_auc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_auc": val_metrics["roc_auc"],
                "epoch_time": epoch_time,
            }
            history.append(epoch_info)

            current_val_auc = val_metrics["roc_auc"]

            if current_val_auc > best_val_auc:
                best_val_auc = current_val_auc
                best_epoch = epoch
                patience_counter = 0

                best_model_path = CHECKPOINT_DIR / "best_model.pth"
                save_checkpoint(best_model_path, model, optimizer, epoch, val_metrics)
                print(f"[SAVE] Đã lưu Best Model (Val ROC-AUC: {best_val_auc:.4f}) tại {best_model_path}")
            else:
                patience_counter += 1
                print(f"[EARLY STOPPING] {patience_counter}/{PATIENCE} epoch không cải thiện ROC-AUC.")

            if patience_counter >= PATIENCE:
                print(f"\n[EARLY STOPPING] Dừng huấn luyện tại epoch {epoch}. Best epoch: {best_epoch} (ROC-AUC: {best_val_auc:.4f})")
                break

        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("HOÀN TẤT HUẤN LUYỆN FINE-TUNING")
        print("=" * 80)
        print(f"Tổng thời gian: {total_time / 60:.2f} phút")
        print(f"Best Epoch: {best_epoch} | Best Val ROC-AUC: {best_val_auc:.4f}")

        history_path = CHECKPOINT_DIR / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
        print(f"[SAVE] Đã lưu lịch sử huấn luyện tại {history_path}")

    except Exception as e:
        print("\n[ERROR] Quá trình huấn luyện bị lỗi.")
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()