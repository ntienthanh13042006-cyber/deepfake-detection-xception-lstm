import os
import json
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from dataset import DeepfakeDataset
from model import SpatialTemporalXceptionBiLSTM


# ============================================================
# 1. CẤU HÌNH
# ============================================================

DATA_ROOT = r"D:\Project\Processed_Data_v2"

TRAIN_CSV = r"D:\Project\splits\train.csv"
VAL_CSV = r"D:\Project\splits\val.csv"
TEST_CSV = r"D:\Project\splits\test.csv"

CHECKPOINT_DIR = r"D:\Project\checkpoints"

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

HISTORY_PATH = os.path.join(
    CHECKPOINT_DIR,
    "training_history.csv"
)

CONFIG_PATH = os.path.join(
    CHECKPOINT_DIR,
    "training_config.json"
)


# ============================================================
# 2. HYPERPARAMETERS
# ============================================================

BATCH_SIZE = 2

# Số epoch tối đa
NUM_EPOCHS = 10

LEARNING_RATE = 1e-4

SEQUENCE_LENGTH = 15

LSTM_HIDDEN_SIZE = 256

LSTM_LAYERS = 2

# Sau bao nhiêu epoch không cải thiện Val AUC thì dừng
EARLY_STOPPING_PATIENCE = 3

RANDOM_SEED = 42

NUM_WORKERS = 0


# ============================================================
# 3. FIX RANDOM SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# 4. DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print("\n[INFO] CUDA khả dụng")

        print(
            f"[INFO] GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        total_vram = (
            torch.cuda.get_device_properties(0)
            .total_memory / 1024**3
        )

        print(
            f"[INFO] VRAM tổng: "
            f"{total_vram:.2f} GB"
        )

    else:

        device = torch.device("cpu")

        print("\n[WARNING] CUDA không khả dụng!")

        print("[INFO] Đang sử dụng CPU.")

    return device


# ============================================================
# 5. WEIGHTED RANDOM SAMPLER
# ============================================================

def create_weighted_sampler(dataset):

    labels = (
        dataset.df["label"]
        .astype(int)
        .to_numpy()
    )

    class_counts = np.bincount(labels)

    if len(class_counts) < 2:

        raise ValueError(
            "Train dataset phải có cả Real và Fake."
        )

    num_real = class_counts[0]

    num_fake = class_counts[1]

    print("\n[INFO] Phân bố Train:")

    print(f"       Real = {num_real}")

    print(f"       Fake = {num_fake}")

    # Trọng số nghịch đảo số lượng
    real_weight = 1.0 / num_real

    fake_weight = 1.0 / num_fake

    sample_weights = np.array(
        [
            real_weight if label == 0
            else fake_weight
            for label in labels
        ],
        dtype=np.float64
    )

    sample_weights = torch.as_tensor(
        sample_weights,
        dtype=torch.double
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


# ============================================================
# 6. CREATE DATALOADERS
# ============================================================

def create_dataloaders():

    print("\n========== KHỞI TẠO DATASET ==========")

    train_dataset = DeepfakeDataset(
        csv_path=TRAIN_CSV,
        data_root=DATA_ROOT
    )

    val_dataset = DeepfakeDataset(
        csv_path=VAL_CSV,
        data_root=DATA_ROOT
    )

    test_dataset = DeepfakeDataset(
        csv_path=TEST_CSV,
        data_root=DATA_ROOT
    )

    train_sampler = create_weighted_sampler(
        train_dataset
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    print("\n========== DATALOADER ==========")

    print(
        f"Train batches: {len(train_loader)}"
    )

    print(
        f"Val batches:   {len(val_loader)}"
    )

    print(
        f"Test batches:  {len(test_loader)}"
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# 7. METRICS
# ============================================================

def calculate_metrics(
    labels,
    probabilities
):

    labels = np.asarray(labels)

    probabilities = np.asarray(
        probabilities
    )

    predictions = (
        probabilities >= 0.5
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

    if len(np.unique(labels)) == 2:

        auc = roc_auc_score(
            labels,
            probabilities
        )

    else:

        auc = float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


# ============================================================
# 8. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    all_labels = []

    all_probs = []

    for batch_idx, (inputs, labels) in enumerate(loader):

        inputs = inputs.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(inputs)

        logits = logits.squeeze(1)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        batch_size = inputs.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        probabilities = torch.sigmoid(
            logits
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        all_probs.extend(
            probabilities.detach()
            .cpu()
            .numpy()
        )

        if (batch_idx + 1) % 20 == 0:

            print(
                f"      Batch "
                f"{batch_idx + 1}/{len(loader)} "
                f"- Loss: {loss.item():.4f}"
            )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_labels,
        all_probs
    )

    return epoch_loss, metrics


# ============================================================
# 9. VALIDATION / TEST
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    all_labels = []

    all_probs = []

    for inputs, labels in loader:

        inputs = inputs.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        logits = model(inputs)

        logits = logits.squeeze(1)

        loss = criterion(
            logits,
            labels
        )

        batch_size = inputs.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        probabilities = torch.sigmoid(
            logits
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_probs.extend(
            probabilities.cpu().numpy()
        )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_labels,
        all_probs
    )

    return (
        epoch_loss,
        metrics,
        all_labels,
        all_probs
    )


# ============================================================
# 10. IN THÔNG TIN METRICS
# ============================================================

def print_metrics(
    title,
    loss,
    metrics
):

    print(
        f"\n========== {title} =========="
    )

    print(
        f"Loss:       {loss:.4f}"
    )

    print(
        f"Accuracy:   {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision:  {metrics['precision']:.4f}"
    )

    print(
        f"Recall:     {metrics['recall']:.4f}"
    )

    print(
        f"F1:         {metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC:    {metrics['auc']:.4f}"
    )


# ============================================================
# 11. SAVE TRAINING CONFIG
# ============================================================

def save_config(device):

    config = {
        "data_root": DATA_ROOT,
        "train_csv": TRAIN_CSV,
        "val_csv": VAL_CSV,
        "test_csv": TEST_CSV,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "sequence_length": SEQUENCE_LENGTH,
        "lstm_hidden_size": LSTM_HIDDEN_SIZE,
        "lstm_layers": LSTM_LAYERS,
        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,
        "random_seed": RANDOM_SEED,
        "num_workers": NUM_WORKERS,
        "device": str(device),
        "model_name":
            "SpatialTemporalXceptionBiLSTM",
        "xception_feature_dim": 2048,
        "label_mapping": {
            "Real": 0,
            "Fake": 1
        },
        "normalization": {
            "mean": [
                0.485,
                0.456,
                0.406
            ],
            "std": [
                0.229,
                0.224,
                0.225
            ]
        }
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(
        f"\n[INFO] Đã lưu config: "
        f"{CONFIG_PATH}"
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 60)

    print(
        "       DEEPFAKE DETECTION"
    )

    print(
        "       OFFICIAL TRAINING"
    )

    print("=" * 60)

    set_seed(RANDOM_SEED)

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    device = get_device()

    save_config(device)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders()

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "\n========== KHỞI TẠO MODEL =========="
    )

    model = SpatialTemporalXceptionBiLSTM(
        sequence_length=SEQUENCE_LENGTH,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS
    )

    model = model.to(device)

    print(
        "[INFO] Model đã chuyển sang device."
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = torch.nn.BCEWithLogitsLoss()

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    print(
        "\n========== TRAIN CONFIG =========="
    )

    print(
        f"Batch size:      {BATCH_SIZE}"
    )

    print(
        f"Max epochs:      {NUM_EPOCHS}"
    )

    print(
        f"Learning rate:   {LEARNING_RATE}"
    )

    print(
        f"Sequence length: {SEQUENCE_LENGTH}"
    )

    print(
        f"LSTM hidden:     {LSTM_HIDDEN_SIZE}"
    )

    print(
        f"LSTM layers:     {LSTM_LAYERS}"
    )

    print(
        f"Early stopping:  "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    print(
        f"Device:          {device}"
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    history = []

    best_val_auc = -float("inf")

    epochs_without_improvement = 0

    best_epoch = 0

    for epoch in range(NUM_EPOCHS):

        print("\n")

        print("=" * 60)

        print(
            f"EPOCH {epoch + 1}/{NUM_EPOCHS}"
        )

        print("=" * 60)

        # ---------------- TRAIN ----------------

        train_loss, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        # ---------------- VALIDATION ----------------

        (
            val_loss,
            val_metrics,
            _,
            _
        ) = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        # ---------------- PRINT ----------------

        print_metrics(
            "TRAIN RESULT",
            train_loss,
            train_metrics
        )

        print_metrics(
            "VALIDATION RESULT",
            val_loss,
            val_metrics
        )

        # ---------------- HISTORY ----------------

        history_row = {
            "epoch": epoch + 1,

            "train_loss": train_loss,
            "train_accuracy":
                train_metrics["accuracy"],
            "train_precision":
                train_metrics["precision"],
            "train_recall":
                train_metrics["recall"],
            "train_f1":
                train_metrics["f1"],
            "train_auc":
                train_metrics["auc"],

            "val_loss": val_loss,
            "val_accuracy":
                val_metrics["accuracy"],
            "val_precision":
                val_metrics["precision"],
            "val_recall":
                val_metrics["recall"],
            "val_f1":
                val_metrics["f1"],
            "val_auc":
                val_metrics["auc"]
        }

        history.append(history_row)

        # ------------------------------------------------
        # SAVE BEST CHECKPOINT
        # ------------------------------------------------

        current_val_auc = val_metrics["auc"]

        if np.isfinite(current_val_auc):

            if current_val_auc > best_val_auc:

                best_val_auc = current_val_auc

                best_epoch = epoch + 1

                epochs_without_improvement = 0

                torch.save(
                    {
                        "epoch": epoch + 1,

                        "model_state_dict":
                            model.state_dict(),

                        "optimizer_state_dict":
                            optimizer.state_dict(),

                        "val_auc":
                            current_val_auc,

                        "val_f1":
                            val_metrics["f1"],

                        "val_accuracy":
                            val_metrics["accuracy"],

                        "val_precision":
                            val_metrics["precision"],

                        "val_recall":
                            val_metrics["recall"],

                        "sequence_length":
                            SEQUENCE_LENGTH,

                        "lstm_hidden_size":
                            LSTM_HIDDEN_SIZE,

                        "lstm_layers":
                            LSTM_LAYERS,

                        "label_mapping": {
                            "Real": 0,
                            "Fake": 1
                        }
                    },
                    BEST_MODEL_PATH
                )

                print(
                    "\n[INFO] ★ BEST MODEL UPDATED ★"
                )

                print(
                    f"[INFO] Epoch: "
                    f"{epoch + 1}"
                )

                print(
                    f"[INFO] Val ROC-AUC: "
                    f"{current_val_auc:.4f}"
                )

                print(
                    f"[INFO] Path: "
                    f"{BEST_MODEL_PATH}"
                )

            else:

                epochs_without_improvement += 1

                print(
                    "\n[INFO] Validation AUC "
                    "không cải thiện."
                )

                print(
                    f"[INFO] Không cải thiện: "
                    f"{epochs_without_improvement}/"
                    f"{EARLY_STOPPING_PATIENCE}"
                )

        # ------------------------------------------------
        # SAVE HISTORY
        # ------------------------------------------------

        history_df = pd.DataFrame(history)

        history_df.to_csv(
            HISTORY_PATH,
            index=False,
            encoding="utf-8-sig"
        )

        # ------------------------------------------------
        # EARLY STOPPING
        # ------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\n[INFO] EARLY STOPPING."
            )

            print(
                "[INFO] Validation ROC-AUC "
                "không cải thiện."
            )

            break

    # ========================================================
    # TRAINING FINISHED
    # ========================================================

    print("\n")

    print("=" * 60)

    print(
        "           TRAINING FINISHED"
    )

    print("=" * 60)

    print(
        f"[INFO] Best epoch: {best_epoch}"
    )

    print(
        f"[INFO] Best Val ROC-AUC: "
        f"{best_val_auc:.4f}"
    )

    print(
        f"[INFO] History: "
        f"{HISTORY_PATH}"
    )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    if not os.path.exists(
        BEST_MODEL_PATH
    ):

        raise FileNotFoundError(
            "Không tìm thấy best_model.pth."
        )

    print(
        "\n========== LOAD BEST MODEL =========="
    )

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"[INFO] Best checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"[INFO] Best validation AUC: "
        f"{checkpoint['val_auc']:.4f}"
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print("\n")

    print("=" * 60)

    print(
        "             FINAL TEST"
    )

    print("=" * 60)

    (
        test_loss,
        test_metrics,
        _,
        _
    ) = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device
    )

    print_metrics(
        "FINAL TEST RESULT",
        test_loss,
        test_metrics
    )

    print("\n")

    print("=" * 60)

    print(
        "       OFFICIAL TRAINING COMPLETED"
    )

    print("=" * 60)

    print(
        f"[INFO] Best model: "
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"[INFO] Training history: "
        f"{HISTORY_PATH}"
    )

    print(
        f"[INFO] Training config: "
        f"{CONFIG_PATH}"
    )


if __name__ == "__main__":
    main()