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
from model_finetune import (
    SpatialTemporalXceptionBiLSTM_FineTune
)


# ============================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ============================================================

PROJECT_DIR = Path(r"D:\Project")

# ------------------------------------------------------------
# Thư mục gốc chứa NPY đã preprocessing
# ------------------------------------------------------------
PROCESSED_ROOT = (
    PROJECT_DIR / "Processed_Data_CelebDF"
)

# ------------------------------------------------------------
# CSV đã tạo từ bước preprocessing
# ------------------------------------------------------------
TRAIN_CSV = (
    PROJECT_DIR
    / "results_celebdf"
    / "processed_train.csv"
)

VAL_CSV = (
    PROJECT_DIR
    / "results_celebdf"
    / "processed_val.csv"
)

# ------------------------------------------------------------
# Thư mục checkpoint RIÊNG cho experiment fine-tuning
# Không ghi đè baseline.
# ------------------------------------------------------------
CHECKPOINT_DIR = (
    PROJECT_DIR
    / "checkpoints_celebdf_finetune"
)


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

# ------------------------------------------------------------
# Differential learning rate
# ------------------------------------------------------------
XCEPTION_LR = 1e-5
LSTM_CLASSIFIER_LR = 1e-4

WEIGHT_DECAY = 1e-4

# ------------------------------------------------------------
# Early stopping
# ------------------------------------------------------------
PATIENCE = 4

# ------------------------------------------------------------
# Random seed
# ------------------------------------------------------------
SEED = 42


# ============================================================
# 4. RANDOM SEED
# ============================================================

def set_seed(seed: int = 42):
    """
    Thiết lập random seed để tăng tính tái lập.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# 5. RESOLVE ĐƯỜNG DẪN NPY
# ============================================================

def resolve_npy_path(raw_path: str) -> Path:
    """
    Chuyển đường dẫn trong CSV thành đường dẫn thực tế.

    CSV của bạn có thể chứa:

    1. Absolute path:
        D:\\Project\\Processed_Data_CelebDF\\Fake\\...

    2. Relative path:
        Fake\\Celeb-synthesis\\id19_id25_0004.npy

    Với trường hợp relative path, sẽ nối với:

        D:\\Project\\Processed_Data_CelebDF
    """

    raw_path = str(raw_path).strip()

    if not raw_path:
        raise ValueError(
            "Đường dẫn NPY trong CSV đang rỗng."
        )

    path_obj = Path(raw_path)

    # --------------------------------------------------------
    # Trường hợp absolute path
    # --------------------------------------------------------
    if path_obj.is_absolute():
        return path_obj

    # --------------------------------------------------------
    # Trường hợp relative path
    # --------------------------------------------------------
    candidate = (
        PROCESSED_ROOT / path_obj
    )

    # --------------------------------------------------------
    # Chuẩn hóa path
    # --------------------------------------------------------
    return candidate


# ============================================================
# 6. DATASET
# ============================================================

class CelebDFNPYDataset(Dataset):
    """
    Dataset đọc NPY đã preprocessing.

    Input NPY:
        (15, 224, 224, 3), uint8

    Output:
        (15, 3, 224, 224), float32

    Label:
        0 = Real
        1 = Fake
    """

    def __init__(self, csv_path: Path):

        # ====================================================
        # Kiểm tra CSV
        # ====================================================

        if not csv_path.exists():

            raise FileNotFoundError(
                f"Không tìm thấy CSV:\n{csv_path}"
            )

        # ====================================================
        # Kiểm tra thư mục dữ liệu
        # ====================================================

        if not PROCESSED_ROOT.exists():

            raise FileNotFoundError(
                "Không tìm thấy thư mục NPY:\n"
                f"{PROCESSED_ROOT}"
            )

        # ====================================================
        # Đọc CSV
        # ====================================================

        self.df = pd.read_csv(csv_path)

        if len(self.df) == 0:

            raise ValueError(
                f"CSV không có dữ liệu:\n"
                f"{csv_path}"
            )

        # ====================================================
        # Tìm cột path
        # ====================================================

        possible_path_columns = [
            "npy_path",
            "path",
            "processed_path",
            "file_path",
        ]

        self.path_column = None

        for column in possible_path_columns:

            if column in self.df.columns:

                self.path_column = column

                break

        if self.path_column is None:

            raise ValueError(
                "Không tìm thấy cột đường dẫn NPY.\n"
                f"Các cột hiện có: "
                f"{list(self.df.columns)}"
            )

        # ====================================================
        # Tìm cột label
        # ====================================================

        possible_label_columns = [
            "label",
            "target",
            "class",
        ]

        self.label_column = None

        for column in possible_label_columns:

            if column in self.df.columns:

                self.label_column = column

                break

        if self.label_column is None:

            raise ValueError(
                "Không tìm thấy cột label.\n"
                f"Các cột hiện có: "
                f"{list(self.df.columns)}"
            )

        # ====================================================
        # Chuẩn hóa label
        # ====================================================

        self.df[self.label_column] = (
            self.df[self.label_column]
            .astype(int)
        )

        # ====================================================
        # Kiểm tra label
        # ====================================================

        unique_labels = set(
            self.df[
                self.label_column
            ].unique().tolist()
        )

        if not unique_labels.issubset({0, 1}):

            raise ValueError(
                f"Label không hợp lệ: "
                f"{unique_labels}\n"
                "Quy ước: 0=Real, 1=Fake."
            )

        # ====================================================
        # Resolve toàn bộ đường dẫn
        # ====================================================

        print("\n" + "=" * 80)
        print(
            f"KIỂM TRA ĐƯỜNG DẪN NPY - "
            f"{csv_path.name}"
        )
        print("=" * 80)

        self.resolved_paths = []

        missing_files = []

        for index, raw_path in enumerate(
            self.df[
                self.path_column
            ].astype(str)
        ):

            try:

                resolved_path = resolve_npy_path(
                    raw_path
                )

            except Exception as e:

                missing_files.append(
                    (
                        index,
                        raw_path,
                        f"Resolve error: {e}"
                    )
                )

                continue

            self.resolved_paths.append(
                resolved_path
            )

            # ------------------------------------------------
            # Kiểm tra tồn tại
            # ------------------------------------------------

            if not resolved_path.exists():

                missing_files.append(
                    (
                        index,
                        raw_path,
                        str(resolved_path)
                    )
                )

        # ====================================================
        # Kiểm tra số lượng path
        # ====================================================

        if len(self.resolved_paths) != len(self.df):

            raise RuntimeError(
                "Số lượng đường dẫn resolve được "
                "không khớp số lượng dòng trong CSV."
            )

        total_files = len(
            self.resolved_paths
        )

        missing_count = len(
            missing_files
        )

        existing_count = (
            total_files - missing_count
        )

        print(
            f"[INFO] Tổng NPY : "
            f"{total_files}"
        )

        print(
            f"[INFO] Tồn tại  : "
            f"{existing_count}"
        )

        print(
            f"[INFO] Thiếu    : "
            f"{missing_count}"
        )

        # ====================================================
        # In một vài path mẫu
        # ====================================================

        print("\n[INFO] Một số path sau khi resolve:")

        for i in range(
            min(3, len(self.resolved_paths))
        ):

            print(
                f"  [{i}] "
                f"{self.resolved_paths[i]}"
            )

        # ====================================================
        # Nếu có file thiếu -> dừng ngay
        # ====================================================

        if missing_count > 0:

            print("\n" + "=" * 80)
            print(
                "DANH SÁCH FILE NPY BỊ THIẾU"
            )
            print("=" * 80)

            for item in missing_files[:20]:

                index, raw_path, resolved = item

                print(
                    f"\n[MISSING] index={index}"
                )

                print(
                    f"  CSV     : {raw_path}"
                )

                print(
                    f"  Resolved: {resolved}"
                )

            if missing_count > 20:

                print(
                    f"\n... còn "
                    f"{missing_count - 20} "
                    "file khác."
                )

            raise FileNotFoundError(
                f"Có {missing_count}/"
                f"{total_files} file NPY "
                "không tồn tại."
            )

        print(
            f"[PASS] Tất cả {total_files} "
            "đường dẫn NPY hợp lệ."
        )

        print("=" * 80)

        # ====================================================
        # Chuẩn bị normalization constants
        # ====================================================

        self.mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32
        ).view(
            1,
            3,
            1,
            1
        )

        self.std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32
        ).view(
            1,
            3,
            1,
            1
        )

    # ========================================================
    # Length
    # ========================================================

    def __len__(self):

        return len(self.df)

    # ========================================================
    # Get item
    # ========================================================

    def __getitem__(self, idx):

        # ----------------------------------------------------
        # Path thực tế
        # ----------------------------------------------------

        npy_path = (
            self.resolved_paths[idx]
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        row = self.df.iloc[idx]

        label = int(
            row[self.label_column]
        )

        # ----------------------------------------------------
        # Đọc NPY
        # ----------------------------------------------------

        try:

            frames = np.load(
                npy_path
            )

        except Exception as e:

            raise RuntimeError(
                "Không thể đọc NPY:\n"
                f"{npy_path}\n"
                f"Lỗi: {e}"
            )

        # ====================================================
        # Kiểm tra shape
        # ====================================================

        expected_shape = (
            SEQUENCE_LENGTH,
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        )

        if frames.shape != expected_shape:

            raise ValueError(
                "Shape NPY không đúng.\n"
                f"File      : {npy_path}\n"
                f"Nhận      : {frames.shape}\n"
                f"Kỳ vọng   : {expected_shape}"
            )

        # ====================================================
        # Kiểm tra dtype
        # ====================================================

        if frames.dtype != np.uint8:

            # ------------------------------------------------
            # Dataset hiện tại dự kiến uint8.
            # Nếu khác thì vẫn chuyển float32 để tránh crash.
            # ------------------------------------------------
            frames = frames.astype(
                np.float32
            )

        else:

            frames = frames.astype(
                np.float32
            )

        # ====================================================
        # [0,255] -> [0,1]
        # ====================================================

        frames = frames / 255.0

        # ====================================================
        # NHWC -> NCHW
        # ====================================================

        frames = torch.from_numpy(
            frames
        )

        frames = frames.permute(
            0,
            3,
            1,
            2
        ).contiguous()

        # ====================================================
        # ImageNet normalization
        # ====================================================

        frames = (
            frames - self.mean
        ) / self.std

        # ====================================================
        # Label tensor
        # ====================================================

        label_tensor = torch.tensor(
            label,
            dtype=torch.float32
        )

        return (
            frames,
            label_tensor
        )


# ============================================================
# 7. METRICS
# ============================================================

def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray
):
    """
    Tính metric tại threshold 0.5.

    Lưu ý:
    Đây chỉ là metric theo dõi trong quá trình training.

    Threshold chính thức sẽ được tối ưu riêng
    trên Validation sau khi training hoàn tất.
    """

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    labels = labels.astype(int)

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

    try:

        auc = roc_auc_score(
            labels,
            probabilities
        )

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

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
):
    """
    Huấn luyện một epoch.
    """

    model.train()

    total_loss = 0.0

    all_labels = []

    all_probs = []

    for batch_idx, (
        inputs,
        labels
    ) in enumerate(loader):

        # ----------------------------------------------------
        # Move to GPU
        # ----------------------------------------------------

        inputs = inputs.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # AMP
        # ----------------------------------------------------

        with autocast(
            enabled=(
                device.type == "cuda"
            )
        ):

            logits = model(
                inputs
            )

            logits = logits.squeeze(
                1
            )

            loss = criterion(
                logits,
                labels
            )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        current_batch_size = (
            inputs.size(0)
        )

        total_loss += (
            loss.item()
            * current_batch_size
        )

        probabilities = torch.sigmoid(
            logits
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_probs.extend(
            probabilities.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (batch_idx + 1) % 500
            == 0
        ):

            print(
                f"    [Train] "
                f"Batch {batch_idx + 1}/"
                f"{len(loader)} | "
                f"Loss = "
                f"{loss.item():.4f}"
            )

    # ========================================================
    # Epoch statistics
    # ========================================================

    epoch_loss = (
        total_loss
        / len(loader.dataset)
    )

    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_probs)
    )

    metrics["loss"] = epoch_loss

    return metrics


# ============================================================
# 9. VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):
    """
    Validation toàn bộ validation set.
    """

    model.eval()

    total_loss = 0.0

    all_labels = []

    all_probs = []

    for inputs, labels in loader:

        # ----------------------------------------------------
        # Move to device
        # ----------------------------------------------------

        inputs = inputs.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        with autocast(
            enabled=(
                device.type == "cuda"
            )
        ):

            logits = model(
                inputs
            )

            logits = logits.squeeze(
                1
            )

            loss = criterion(
                logits,
                labels
            )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        current_batch_size = (
            inputs.size(0)
        )

        total_loss += (
            loss.item()
            * current_batch_size
        )

        probabilities = torch.sigmoid(
            logits
        )

        all_labels.extend(
            labels
            .cpu()
            .numpy()
            .tolist()
        )

        all_probs.extend(
            probabilities
            .cpu()
            .numpy()
            .tolist()
        )

    # ========================================================
    # Validation statistics
    # ========================================================

    epoch_loss = (
        total_loss
        / len(loader.dataset)
    )

    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_probs)
    )

    metrics["loss"] = epoch_loss

    return metrics


# ============================================================
# 10. BUILD OPTIMIZER
# ============================================================

def build_optimizer(model):
    """
    Differential learning rate:

        Xception      = 1e-5
        Bi-LSTM       = 1e-4
        Classifier    = 1e-4
    """

    xception_params = []

    other_params = []

    for name, param in model.named_parameters():

        if not param.requires_grad:

            continue

        if name.startswith(
            "xception."
        ):

            xception_params.append(
                param
            )

        else:

            other_params.append(
                param
            )

    if len(xception_params) == 0:

        raise RuntimeError(
            "Không có parameter Xception "
            "nào được fine-tune."
        )

    if len(other_params) == 0:

        raise RuntimeError(
            "Không có parameter "
            "Bi-LSTM/classifier để train."
        )

    optimizer = torch.optim.Adam(
        [
            {
                "params":
                    xception_params,

                "lr":
                    XCEPTION_LR,
            },
            {
                "params":
                    other_params,

                "lr":
                    LSTM_CLASSIFIER_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # In thông tin optimizer
    # ========================================================

    print("\n" + "=" * 80)
    print("OPTIMIZER")
    print("=" * 80)

    print(
        "Xception trainable params : "
        f"{sum(p.numel() for p in xception_params):,}"
    )

    print(
        "LSTM + classifier params  : "
        f"{sum(p.numel() for p in other_params):,}"
    )

    print(
        f"Xception LR               : "
        f"{XCEPTION_LR}"
    )

    print(
        f"LSTM/classifier LR        : "
        f"{LSTM_CLASSIFIER_LR}"
    )

    print(
        f"Weight decay              : "
        f"{WEIGHT_DECAY}"
    )

    print("=" * 80)

    return optimizer


# ============================================================
# 11. SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    val_metrics,
):
    """
    Lưu best checkpoint.
    """

    checkpoint = {
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "val_metrics":
            val_metrics,

        "config": {
            "sequence_length":
                SEQUENCE_LENGTH,

            "image_size":
                IMAGE_SIZE,

            "batch_size":
                BATCH_SIZE,

            "max_epochs":
                MAX_EPOCHS,

            "xception_lr":
                XCEPTION_LR,

            "lstm_classifier_lr":
                LSTM_CLASSIFIER_LR,

            "weight_decay":
                WEIGHT_DECAY,

            "patience":
                PATIENCE,

            "seed":
                SEED,

            "fine_tuned": [
                "xception.block12",
                "xception.conv3",
                "xception.bn3",
                "xception.conv4",
                "xception.bn4",
            ],

            "frozen":
                "xception.block1 -> block11",

            "processed_root":
                str(PROCESSED_ROOT),
        },
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    try:

        print("=" * 80)
        print(
            "G7.4 - TRAIN FINE-TUNING "
            "XCEPTION + BI-LSTM"
        )
        print("=" * 80)

        # ====================================================
        # Seed
        # ====================================================

        set_seed(SEED)

        # ====================================================
        # Device
        # ====================================================

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"[INFO] Device: {device}"
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
        # Kiểm tra path
        # ====================================================

        print("\n" + "=" * 80)
        print("KIỂM TRA ĐƯỜNG DẪN CHÍNH")
        print("=" * 80)

        print(
            f"[INFO] Project      : "
            f"{PROJECT_DIR}"
        )

        print(
            f"[INFO] Processed    : "
            f"{PROCESSED_ROOT}"
        )

        print(
            f"[INFO] Train CSV    : "
            f"{TRAIN_CSV}"
        )

        print(
            f"[INFO] Val CSV      : "
            f"{VAL_CSV}"
        )

        print(
            f"[INFO] Checkpoint   : "
            f"{CHECKPOINT_DIR}"
        )

        if not PROCESSED_ROOT.exists():

            raise FileNotFoundError(
                "Thư mục Processed_Data_CelebDF "
                "không tồn tại:\n"
                f"{PROCESSED_ROOT}"
            )

        # ====================================================
        # Tạo checkpoint directory
        # ====================================================

        CHECKPOINT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # ====================================================
        # Dataset
        # ====================================================

        print("\n[INFO] Loading dataset...")

        train_dataset = (
            CelebDFNPYDataset(
                TRAIN_CSV
            )
        )

        val_dataset = (
            CelebDFNPYDataset(
                VAL_CSV
            )
        )

        print(
            f"\n[INFO] Train samples: "
            f"{len(train_dataset)}"
        )

        print(
            f"[INFO] Val samples  : "
            f"{len(val_dataset)}"
        )

        # ====================================================
        # Class distribution
        # ====================================================

        train_labels = (
            train_dataset.df[
                train_dataset.label_column
            ]
            .astype(int)
            .values
        )

        real_count = int(
            np.sum(
                train_labels == 0
            )
        )

        fake_count = int(
            np.sum(
                train_labels == 1
            )
        )

        print("\n" + "=" * 80)
        print("TRAIN CLASS DISTRIBUTION")
        print("=" * 80)

        print(
            f"Real: {real_count}"
        )

        print(
            f"Fake: {fake_count}"
        )

        print("=" * 80)

        # ====================================================
        # WeightedRandomSampler
        # ====================================================

        class_counts = np.bincount(
            train_labels,
            minlength=2
        )

        if np.any(
            class_counts == 0
        ):

            raise ValueError(
                "Train dataset phải có cả "
                "Real và Fake."
            )

        class_weights = (
            1.0 / class_counts
        )

        sample_weights = (
            class_weights[
                train_labels
            ]
        )

        sample_weights = (
            torch.as_tensor(
                sample_weights,
                dtype=torch.double
            )
        )

        sampler = (
            WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(
                    sample_weights
                ),
                replacement=True,
            )
        )

        # ====================================================
        # DataLoader
        # ====================================================

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=(
                torch.cuda.is_available()
            ),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(
                torch.cuda.is_available()
            ),
        )

        print("\n" + "=" * 80)
        print("DATALOADER")
        print("=" * 80)

        print(
            f"Train batches: "
            f"{len(train_loader)}"
        )

        print(
            f"Val batches  : "
            f"{len(val_loader)}"
        )

        print(
            f"Batch size   : "
            f"{BATCH_SIZE}"
        )

        print(
            f"Workers      : "
            f"{NUM_WORKERS}"
        )

        print("=" * 80)

        # ====================================================
        # Model
        # ====================================================

        print(
            "\n[INFO] Khởi tạo model fine-tune..."
        )

        model = (
            SpatialTemporalXceptionBiLSTM_FineTune(
                sequence_length=15,
                lstm_hidden_size=256,
                lstm_layers=2,
                dropout=0.5,
            )
        )

        model = model.to(
            device
        )

        # ====================================================
        # Parameter statistics
        # ====================================================

        total_params = sum(
            p.numel()
            for p in model.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        frozen_params = (
            total_params
            - trainable_params
        )

        print("\n" + "=" * 80)
        print("MODEL PARAMETERS")
        print("=" * 80)

        print(
            f"Total parameters     : "
            f"{total_params:,}"
        )

        print(
            f"Trainable parameters : "
            f"{trainable_params:,}"
        )

        print(
            f"Frozen parameters    : "
            f"{frozen_params:,}"
        )

        print(
            f"Trainable ratio      : "
            f"{trainable_params / total_params * 100:.2f}%"
        )

        print("=" * 80)

        # ====================================================
        # Kiểm tra fine-tuning
        # ====================================================

        print("\n" + "=" * 80)
        print("KIỂM TRA FINE-TUNING")
        print("=" * 80)

        block11_trainable = any(
            p.requires_grad
            for p in
            model.xception.block11.parameters()
        )

        block12_trainable = any(
            p.requires_grad
            for p in
            model.xception.block12.parameters()
        )

        conv3_trainable = any(
            p.requires_grad
            for p in
            model.xception.conv3.parameters()
        )

        conv4_trainable = any(
            p.requires_grad
            for p in
            model.xception.conv4.parameters()
        )

        print(
            f"block11 trainable: "
            f"{block11_trainable}"
        )

        print(
            f"block12 trainable: "
            f"{block12_trainable}"
        )

        print(
            f"conv3 trainable   : "
            f"{conv3_trainable}"
        )

        print(
            f"conv4 trainable   : "
            f"{conv4_trainable}"
        )

        # ----------------------------------------------------
        # Điều kiện bắt buộc
        # ----------------------------------------------------

        if block11_trainable:

            raise RuntimeError(
                "LỖI: block11 phải được FROZEN."
            )

        if not block12_trainable:

            raise RuntimeError(
                "LỖI: block12 phải TRAINABLE."
            )

        if not conv3_trainable:

            raise RuntimeError(
                "LỖI: conv3 phải TRAINABLE."
            )

        if not conv4_trainable:

            raise RuntimeError(
                "LỖI: conv4 phải TRAINABLE."
            )

        print(
            "[PASS] Fine-tuning configuration hợp lệ."
        )

        # ====================================================
        # Loss
        # ====================================================

        criterion = (
            nn.BCEWithLogitsLoss()
        )

        # ====================================================
        # Optimizer
        # ====================================================

        optimizer = (
            build_optimizer(model)
        )

        # ====================================================
        # AMP
        # ====================================================

        scaler = GradScaler(
            enabled=(
                device.type == "cuda"
            )
        )

        # ====================================================
        # History
        # ====================================================

        history = []

        best_val_auc = -np.inf

        best_epoch = 0

        patience_counter = 0

        # ====================================================
        # Training loop
        # ====================================================

        for epoch in range(
            1,
            MAX_EPOCHS + 1
        ):

            print("\n")
            print("=" * 80)
            print(
                f"EPOCH {epoch}/{MAX_EPOCHS}"
            )
            print("=" * 80)

            epoch_start = time.time()

            # ------------------------------------------------
            # TRAIN
            # ------------------------------------------------

            train_metrics = (
                train_one_epoch(
                    model=model,
                    loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                )
            )

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            val_metrics = (
                validate(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                )
            )

            epoch_time = (
                time.time()
                - epoch_start
            )

            # =================================================
            # Print kết quả
            # =================================================

            print("\n[RESULT]")

            print(
                f"Train Loss      : "
                f"{train_metrics['loss']:.4f}"
            )

            print(
                f"Train Accuracy  : "
                f"{train_metrics['accuracy']:.4f}"
            )

            print(
                f"Train Precision : "
                f"{train_metrics['precision']:.4f}"
            )

            print(
                f"Train Recall    : "
                f"{train_metrics['recall']:.4f}"
            )

            print(
                f"Train F1        : "
                f"{train_metrics['f1']:.4f}"
            )

            print(
                f"Train ROC-AUC   : "
                f"{train_metrics['roc_auc']:.4f}"
            )

            print("-" * 80)

            print(
                f"Val Loss        : "
                f"{val_metrics['loss']:.4f}"
            )

            print(
                f"Val Accuracy    : "
                f"{val_metrics['accuracy']:.4f}"
            )

            print(
                f"Val Precision   : "
                f"{val_metrics['precision']:.4f}"
            )

            print(
                f"Val Recall      : "
                f"{val_metrics['recall']:.4f}"
            )

            print(
                f"Val F1          : "
                f"{val_metrics['f1']:.4f}"
            )

            print(
                f"Val ROC-AUC     : "
                f"{val_metrics['roc_auc']:.4f}"
            )

            print(
                f"Time            : "
                f"{epoch_time / 60:.2f} min"
            )

            # =================================================
            # Lưu history
            # =================================================

            history_row = {
                "epoch":
                    epoch,

                "train_loss":
                    train_metrics["loss"],

                "train_accuracy":
                    train_metrics["accuracy"],

                "train_precision":
                    train_metrics["precision"],

                "train_recall":
                    train_metrics["recall"],

                "train_f1":
                    train_metrics["f1"],

                "train_roc_auc":
                    train_metrics["roc_auc"],

                "val_loss":
                    val_metrics["loss"],

                "val_accuracy":
                    val_metrics["accuracy"],

                "val_precision":
                    val_metrics["precision"],

                "val_recall":
                    val_metrics["recall"],

                "val_f1":
                    val_metrics["f1"],

                "val_roc_auc":
                    val_metrics["roc_auc"],

                "time_min":
                    epoch_time / 60,
            }

            history.append(
                history_row
            )

            history_df = (
                pd.DataFrame(history)
            )

            history_df.to_csv(
                CHECKPOINT_DIR
                / "training_history.csv",
                index=False
            )

            # =================================================
            # Best checkpoint theo Validation ROC-AUC
            # =================================================

            current_auc = (
                val_metrics["roc_auc"]
            )

            if (
                not np.isnan(current_auc)
                and current_auc > best_val_auc
            ):

                best_val_auc = (
                    current_auc
                )

                best_epoch = (
                    epoch
                )

                patience_counter = 0

                save_checkpoint(
                    CHECKPOINT_DIR
                    / "best_model.pth",

                    model,

                    optimizer,

                    epoch,

                    val_metrics,
                )

                print("\n[BEST]")

                print(
                    "Validation "
                    f"ROC-AUC = "
                    f"{best_val_auc:.4f}"
                )

                print(
                    "[SAVE] "
                    f"{CHECKPOINT_DIR / 'best_model.pth'}"
                )

            else:

                patience_counter += 1

                print(
                    "\n[EARLY STOP] "
                    f"counter = "
                    f"{patience_counter}/"
                    f"{PATIENCE}"
                )

            # =================================================
            # Early stopping
            # =================================================

            if (
                patience_counter
                >= PATIENCE
            ):

                print(
                    "\n[INFO] "
                    "Early stopping."
                )

                break

        # ====================================================
        # Lưu config
        # ====================================================

        config = {
            "experiment":
                "Celeb-DF-v2 "
                "Partial Fine-tuning",

            "dataset":
                "Celeb-DF v2",

            "sequence_length":
                SEQUENCE_LENGTH,

            "image_size":
                IMAGE_SIZE,

            "batch_size":
                BATCH_SIZE,

            "max_epochs":
                MAX_EPOCHS,

            "xception_lr":
                XCEPTION_LR,

            "lstm_classifier_lr":
                LSTM_CLASSIFIER_LR,

            "weight_decay":
                WEIGHT_DECAY,

            "patience":
                PATIENCE,

            "seed":
                SEED,

            "fine_tuned": [
                "xception.block12",
                "xception.conv3",
                "xception.bn3",
                "xception.conv4",
                "xception.bn4",
            ],

            "frozen":
                "xception.block1 -> block11",

            "processed_root":
                str(PROCESSED_ROOT),

            "train_csv":
                str(TRAIN_CSV),

            "val_csv":
                str(VAL_CSV),

            "best_epoch":
                best_epoch,

            "best_val_roc_auc":
                float(best_val_auc),
        }

        config_path = (
            CHECKPOINT_DIR
            / "training_config.json"
        )

        with open(
            config_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                config,
                file,
                indent=4,
                ensure_ascii=False
            )

        # ====================================================
        # Final summary
        # ====================================================

        print("\n")
        print("=" * 80)
        print(
            "G7.4 TRAINING HOÀN TẤT"
        )
        print("=" * 80)

        print(
            f"Best Epoch      : "
            f"{best_epoch}"
        )

        print(
            f"Best Val AUC    : "
            f"{best_val_auc:.4f}"
        )

        print(
            f"Checkpoint      : "
            f"{CHECKPOINT_DIR / 'best_model.pth'}"
        )

        print(
            f"History         : "
            f"{CHECKPOINT_DIR / 'training_history.csv'}"
        )

        print(
            f"Config          : "
            f"{config_path}"
        )

        print("=" * 80)

        print(
            "\n[PASS] Training fine-tuning "
            "đã hoàn tất."
        )

        print(
            "[NEXT] Chuyển sang G7.5: "
            "Validation threshold optimization."
        )

    # ========================================================
    # Ctrl+C
    # ========================================================

    except KeyboardInterrupt:

        print(
            "\n[INFO] "
            "Training bị dừng bởi người dùng."
        )

        sys.exit(1)

    # ========================================================
    # Exception
    # ========================================================

    except Exception as e:

        print(
            "\n[ERROR] "
            "Training thất bại."
        )

        print(
            f"[ERROR] Chi tiết: {e}"
        )

        import traceback

        traceback.print_exc()

        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()