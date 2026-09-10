import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DeepfakeDataset(Dataset):
    """
    Dataset đọc danh sách video từ file CSV split.

    CSV có các cột:
    relative_path,file_name,method,label,target_id,source_id,group_id
    """

    def __init__(self, csv_path, data_root):
        self.csv_path = csv_path
        self.data_root = data_root

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Không tìm thấy file CSV: {csv_path}"
            )

        if not os.path.isdir(data_root):
            raise FileNotFoundError(
                f"Không tìm thấy thư mục dữ liệu: {data_root}"
            )

        self.df = pd.read_csv(csv_path)

        required_columns = [
            "relative_path",
            "file_name",
            "method",
            "label",
            "target_id",
            "source_id",
            "group_id"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"CSV thiếu các cột: {missing_columns}"
            )

        if len(self.df) == 0:
            raise ValueError(
                f"CSV không có dữ liệu: {csv_path}"
            )

        # Chuẩn hóa đường dẫn
        self.df["relative_path"] = (
            self.df["relative_path"]
            .astype(str)
            .str.replace("\\", os.sep)
            .str.replace("/", os.sep)
        )

        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        print(f"\n[INFO] Dataset: {os.path.basename(csv_path)}")
        print(f"[INFO] Số mẫu: {len(self.df)}")

        real_count = int((self.df["label"] == 0).sum())
        fake_count = int((self.df["label"] == 1).sum())

        print(f"[INFO] Real: {real_count}")
        print(f"[INFO] Fake: {fake_count}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        relative_path = row["relative_path"]
        npy_path = os.path.join(
            self.data_root,
            relative_path
        )

        if not os.path.exists(npy_path):
            raise FileNotFoundError(
                f"Không tìm thấy file .npy:\n{npy_path}"
            )

        sequence = np.load(npy_path)

        # Kiểm tra shape sau preprocessing
        expected_shape = (15, 224, 224, 3)

        if sequence.shape != expected_shape:
            raise ValueError(
                f"Shape không đúng tại {npy_path}: "
                f"{sequence.shape}, "
                f"mong đợi {expected_shape}"
            )

        # uint8 -> float32 [0, 1]
        sequence = torch.from_numpy(
            sequence.copy()
        ).float() / 255.0

        # NHWC -> NCHW
        sequence = sequence.permute(0, 3, 1, 2)

        # Normalize từng frame
        normalized_frames = []

        for frame in sequence:
            frame = self.transform(frame)
            normalized_frames.append(frame)

        sequence = torch.stack(normalized_frames)

        label = torch.tensor(
            float(row["label"]),
            dtype=torch.float32
        )

        return sequence, label


if __name__ == "__main__":

    DATA_ROOT = r"D:\Project\Processed_Data_v2"
    TRAIN_CSV = r"D:\Project\splits\train.csv"
    VAL_CSV = r"D:\Project\splits\val.csv"
    TEST_CSV = r"D:\Project\splits\test.csv"

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

    print("\n========== KIỂM TRA DATASET ==========")

    for name, dataset in [
        ("Train", train_dataset),
        ("Validation", val_dataset),
        ("Test", test_dataset)
    ]:

        sequence, label = dataset[0]

        print(f"\n{name}:")
        print(f"  Số mẫu: {len(dataset)}")
        print(f"  Sequence shape: {sequence.shape}")
        print(f"  Dtype: {sequence.dtype}")
        print(f"  Label: {label.item()}")

    print("\n[OK] Dataset kiểm tra thành công.")