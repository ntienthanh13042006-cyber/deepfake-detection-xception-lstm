"""
preprocess_celebdf.py

Mục đích:
- Tiền xử lý Celeb-DF v2 theo các split đã tạo:
    train.csv
    val.csv
    test.csv

Pipeline:
    MP4 → Sample 15 frames → OpenCV BGR → RGB → MTCNN → Face crop 224x224 → Xử lý frame không phát hiện được mặt → Lưu .npy

Output mỗi video:
    shape = (15, 224, 224, 3)
    dtype = uint8

Chức năng:
- Có thể chạy thử N video.
- Có thể chạy tiếp sau khi bị gián đoạn.
- Kiểm tra file .npy đã tồn tại trước khi xử lý lại.
- Xuất manifest và report.
"""

from pathlib import Path
from collections import Counter, defaultdict
import time
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from facenet_pytorch import MTCNN

# ============================================================
# 1. CẤU HÌNH
# ============================================================
DATASET_ROOT = Path(r"D:\TTCS_nhom74\Celeb-DF-v2")
SPLIT_DIR = Path(r"D:\Project\splits_celebdf")
OUTPUT_ROOT = Path(r"D:\Project\Processed_Data_CelebDF")
RESULTS_DIR = Path(r"D:\Project\results_celebdf")

SEQUENCE_LENGTH = 15
IMAGE_SIZE = 224
MARGIN = 20

MAX_VIDEOS = None

SPLIT_FILES = {
    "train": SPLIT_DIR / "train.csv",
    "val": SPLIT_DIR / "val.csv",
    "test": SPLIT_DIR / "test.csv",
}

# ============================================================
# 2. KIỂM TRA DEVICE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 3. HÀM KHỞI TẠO MTCNN
# ============================================================
def create_mtcnn():
    """Khởi tạo MTCNN."""
    print("\n[INFO] Khởi tạo MTCNN...")
    print(f"[INFO] Device: {DEVICE}")
    mtcnn = MTCNN(
        image_size=IMAGE_SIZE,
        margin=MARGIN,
        keep_all=False,
        select_largest=True,
        post_process=False,
        device=DEVICE,
    )
    print("[OK] MTCNN sẵn sàng.")
    return mtcnn

# ============================================================
# 4. LOAD SPLIT CSV
# ============================================================
def load_split_files():
    """Đọc train.csv / val.csv / test.csv."""
    dataframes = []
    required_columns = {"relative_path", "file_name", "source", "label"}
    for split_name, csv_path in SPLIT_FILES.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Không tìm thấy:\n{csv_path}")
        df = pd.read_csv(csv_path)
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path.name} thiếu cột: {sorted(missing)}")
        df = df.copy()
        df["split"] = split_name
        dataframes.append(df)
        print(f"[OK] {split_name:<5}: {len(df)} video")
    all_df = pd.concat(dataframes, ignore_index=True)
    return all_df

# ============================================================
# 5. CHUYỂN ĐƯỜNG DẪN OUTPUT
# ============================================================
def get_output_path(row):
    """Xác định nơi lưu .npy."""
    source = str(row["source"])
    file_name = str(row["file_name"])
    label = int(row["label"])
    class_name = "Real" if label == 0 else "Fake"
    output_dir = OUTPUT_ROOT / class_name / source
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_name = Path(file_name).stem + ".npy"
    return output_dir / npy_name

# ============================================================
# 6. CHUYỂN ĐƯỜNG DẪN INPUT
# ============================================================
def get_input_path(row):
    """Lấy đường dẫn MP4 từ relative_path."""
    relative_path = str(row["relative_path"]).replace("\\", "/")
    return DATASET_ROOT / Path(relative_path)

# ============================================================
# 7. SAMPLE FRAME
# ============================================================
def sample_frames(video_path: Path, sequence_length: int):
    """Lấy đều sequence_length frame từ video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video:\n{video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError("Video không có frame.")

        if total_frames >= sequence_length:
            frame_indices = np.linspace(0, total_frames - 1, sequence_length).astype(int)
        else:
            frame_indices = np.arange(total_frames)

        frames_rgb = []
        for index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = cap.read()
            if not success:
                frames_rgb.append(None)
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_rgb.append(frame_rgb)

        valid_frames = [f for f in frames_rgb if f is not None]
        if not valid_frames:
            raise RuntimeError("Không đọc được frame nào.")

        while len(frames_rgb) < sequence_length:
            frames_rgb.append(valid_frames[-1])

        frames_rgb = frames_rgb[:sequence_length]

        frame_indices = list(frame_indices)
        while len(frame_indices) < sequence_length:
            frame_indices.append(frame_indices[-1])
        frame_indices = frame_indices[:sequence_length]

        return frames_rgb, frame_indices, total_frames
    finally:
        cap.release()

# ============================================================
# 8. CHUYỂN KẾT QUẢ MTCNN SANG NUMPY
# ============================================================
def face_to_numpy(face):
    """Chuyển output từ MTCNN về: (224, 224, 3), uint8"""
    if face is None:
        return None

    if torch.is_tensor(face):
        face = face.detach().cpu()
        if face.ndim == 3 and face.shape[0] == 3:
            face = face.permute(1, 2, 0)
        face = face.numpy()
    else:
        face = np.asarray(face)

    if face.ndim != 3:
        return None

    if np.issubdtype(face.dtype, np.floating):
        max_value = float(np.max(face))
        if max_value <= 1.0:
            face = face * 255.0
        face = np.clip(face, 0, 255).astype(np.uint8)
    else:
        face = np.clip(face, 0, 255).astype(np.uint8)

    if face.shape[0] != IMAGE_SIZE or face.shape[1] != IMAGE_SIZE:
        face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    if face.shape[2] == 4:
        face = face[:, :, :3]
    if face.shape[2] != 3:
        return None

    return face

# ============================================================
# 9. FILL CÁC FRAME KHÔNG PHÁT HIỆN ĐƯỢC MẶT
# ============================================================
def fill_missing_faces(faces):
    """
    faces là list gồm: np.ndarray hoặc None
    Chiến lược:
    1. Nếu có frame trước đó -> dùng frame trước.
    2. Nếu frame đầu bị thiếu -> tìm frame hợp lệ gần nhất.
    3. Cuối cùng đảm bảo đủ 15 frame.
    """
    valid_indices = [i for i, face in enumerate(faces) if face is not None]
    if not valid_indices:
        return None

    filled = list(faces)
    for i in range(len(filled)):
        if filled[i] is not None:
            continue
        nearest_index = min(valid_indices, key=lambda x: abs(x - i))
        filled[i] = filled[nearest_index].copy()

    return filled

# ============================================================
# 10. PREPROCESS MỘT VIDEO
# ============================================================
def process_video(video_path: Path, output_path: Path, mtcnn):
    """Tiền xử lý một video."""
    try:
        if not video_path.exists():
            raise FileNotFoundError("File video không tồn tại.")

        frames, frame_indices, total_frames = sample_frames(video_path, SEQUENCE_LENGTH)

        faces = []
        detection_count = 0
        for frame in frames:
            if frame is None:
                faces.append(None)
                continue

            pil_image = Image.fromarray(frame)
            face = mtcnn(pil_image)
            if face is None:
                faces.append(None)
                continue

            face_np = face_to_numpy(face)
            if face_np is None:
                faces.append(None)
                continue

            faces.append(face_np)
            detection_count += 1

        faces = fill_missing_faces(faces)
        if faces is None:
            raise RuntimeError("Không phát hiện được khuôn mặt trong toàn bộ 15 frame.")

        sequence = np.stack(faces, axis=0)
        expected_shape = (SEQUENCE_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 3)
        if sequence.shape != expected_shape:
            raise RuntimeError(f"Shape không đúng: {sequence.shape}, expected={expected_shape}")

        sequence = sequence.astype(np.uint8)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(".tmp.npy")
        np.save(temp_path, sequence, allow_pickle=False)

        if output_path.exists():
            output_path.unlink()
        temp_path.replace(output_path)

        return {
            "success": True,
            "detection_count": detection_count,
            "total_frames": total_frames,
            "frame_indices": str(frame_indices),
            "error": "",
        }

    except Exception as e:
        return {
            "success": False,
            "detection_count": detection_count if "detection_count" in locals() else 0,
            "total_frames": total_frames if "total_frames" in locals() else 0,
            "frame_indices": "",
            "error": f"{type(e).__name__}: {e}",
        }

# ============================================================
# 11. KIỂM TRA FILE NPY ĐÃ CÓ
# ============================================================
def validate_existing_npy(path: Path):
    """Kiểm tra file NPY đã có."""
    try:
        if not path.exists():
            return False
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        if data.shape != (SEQUENCE_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 3):
            return False
        if data.dtype != np.uint8:
            return False
        return True
    except Exception:
        return False

# ============================================================
# 12. MAIN
# ============================================================
def main():
    start_time = time.time()
    print("=" * 80)
    print("PREPROCESS CELEB-DF v2")
    print("=" * 80)
    print(f"\nDataset : {DATASET_ROOT}")
    print(f"Output  : {OUTPUT_ROOT}")
    print(f"Device  : {DEVICE}")
    print(f"Sequence: {SEQUENCE_LENGTH}")
    print(f"Image   : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Margin  : {MARGIN}")
    print(f"Max video: {MAX_VIDEOS}")

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset:\n{DATASET_ROOT}")
    if not SPLIT_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy split:\n{SPLIT_DIR}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Đọc split...")
    df = load_split_files()
    if len(df) != 6529:
        raise RuntimeError(f"Tổng split phải bằng 6529, nhưng hiện tại = {len(df)}")
    print(f"[OK] Tổng số video: {len(df)}")

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    if MAX_VIDEOS is not None:
        if MAX_VIDEOS <= 0:
            raise ValueError("MAX_VIDEOS phải > 0 hoặc None.")
        process_df = df.head(MAX_VIDEOS).copy()
    else:
        process_df = df.copy()

    print(f"\n[INFO] Sẽ xử lý: {len(process_df)} video")

    mtcnn = create_mtcnn()

    results = []
    counters = Counter()
    source_counters = defaultdict(Counter)

    print("\n" + "=" * 80)
    print("BẮT ĐẦU PREPROCESSING")
    print("=" * 80)

    for position, (_, row) in enumerate(process_df.iterrows(), start=1):
        split_name = str(row["split"])
        source = str(row["source"])
        label = int(row["label"])
        relative_path = str(row["relative_path"])
        video_path = get_input_path(row)
        output_path = get_output_path(row)

        print(f"\n[{position}/{len(process_df)}] [{split_name.upper():5}] [{source:16}] {row['file_name']}")

        if validate_existing_npy(output_path):
            print("  [SKIP] NPY đã tồn tại và hợp lệ.")
            result = {
                "relative_path": relative_path,
                "file_name": row["file_name"],
                "source": source,
                "label": label,
                "split": split_name,
                "video_path": str(video_path),
                "processed_path": str(output_path.relative_to(OUTPUT_ROOT).as_posix()),
                "status": "SKIPPED_EXISTING",
                "success": True,
                "detection_count": -1,
                "total_frames": -1,
                "error": "",
            }
            counters["skipped"] += 1
            source_counters[source]["skipped"] += 1
            results.append(result)
            continue

        result_info = process_video(video_path, output_path, mtcnn)

        if result_info["success"]:
            status = "SUCCESS"
            print("  [OK] Thành công")
            print(f"  Frames: {result_info['total_frames']}")
            print(f"  Face detected: {result_info['detection_count']}/{SEQUENCE_LENGTH}")
            print(f"  Output: {output_path}")
            counters["success"] += 1
            source_counters[source]["success"] += 1
        else:
            status = "FAILED"
            print(f"  [ERROR] {result_info['error']}")
            counters["failed"] += 1
            source_counters[source]["failed"] += 1

        result = {
            "relative_path": relative_path,
            "file_name": row["file_name"],
            "source": source,
            "label": label,
            "split": split_name,
            "video_path": str(video_path),
            "processed_path": str(output_path.relative_to(OUTPUT_ROOT).as_posix()),
            "status": status,
            "success": result_info["success"],
            "detection_count": result_info["detection_count"],
            "total_frames": result_info["total_frames"],
            "error": result_info["error"],
        }
        results.append(result)

        if position % 10 == 0 or position == len(process_df):
            success_count = counters["success"]
            failed_count = counters["failed"]
            skipped_count = counters["skipped"]
            print("\n" + "-" * 80)
            print(f"TIẾN ĐỘ: {position}/{len(process_df)}")
            print(f"Success: {success_count}")
            print(f"Skipped: {skipped_count}")
            print(f"Failed : {failed_count}")
            print("-" * 80)

    result_df = pd.DataFrame(results)
    manifest_file = RESULTS_DIR / "celebdf_preprocessing_manifest.csv"
    result_df.to_csv(manifest_file, index=False, encoding="utf-8-sig")

    if MAX_VIDEOS is None:
        print("\n[INFO] Tạo processed split CSV...")
        manifest_mapping = result_df[["relative_path", "processed_path", "status"]].drop_duplicates(subset=["relative_path"])
        merged_df = df.merge(manifest_mapping[["relative_path", "processed_path", "status"]], on="relative_path", how="left")

        for split_name in ["train", "val", "test"]:
            split_df = merged_df[merged_df["split"] == split_name].copy()
            output_csv = RESULTS_DIR / f"processed_{split_name}.csv"
            split_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"[OK] {output_csv}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("KẾT QUẢ PREPROCESSING")
    print("=" * 80)
    print(f"\nTổng xử lý : {len(process_df)}")
    print(f"Success    : {counters['success']}")
    print(f"Skipped    : {counters['skipped']}")
    print(f"Failed     : {counters['failed']}")
    print(f"Thời gian  : {elapsed / 60:.2f} phút")

    print("\nTHỐNG KÊ THEO SOURCE")
    print("-" * 80)
    for source in sorted(source_counters.keys()):
        values = source_counters[source]
        print(f"{source:<20} Success={values['success']:<5} Skip={values['skipped']:<5} Failed={values['failed']}")

    report_file = RESULTS_DIR / "celebdf_preprocessing_report.txt"
    with report_file.open("w", encoding="utf-8") as f:
        f.write("CELEB-DF v2 PREPROCESSING REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: {DATASET_ROOT}\n")
        f.write(f"Output: {OUTPUT_ROOT}\n")
        f.write(f"Sequence length: {SEQUENCE_LENGTH}\n")
        f.write(f"Image size: {IMAGE_SIZE}\n")
        f.write(f"Device: {DEVICE}\n\n")
        f.write(f"Processed: {len(process_df)}\n")
        f.write(f"Success: {counters['success']}\n")
        f.write(f"Skipped: {counters['skipped']}\n")
        f.write(f"Failed: {counters['failed']}\n")
        f.write(f"Elapsed minutes: {elapsed / 60:.2f}\n\n")
        f.write("SOURCE STATISTICS\n")
        f.write("-" * 80 + "\n")
        for source in sorted(source_counters.keys()):
            values = source_counters[source]
            f.write(f"{source}: success={values['success']}, skipped={values['skipped']}, failed={values['failed']}\n")

    print(f"\n[INFO] Manifest:\n       {manifest_file}")
    print(f"[INFO] Report:\n       {report_file}")

if __name__ == "__main__":
    main()