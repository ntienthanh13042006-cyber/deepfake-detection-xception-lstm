"""
G8.1 - KIỂM TRA ĐỒNG BỘ PREPROCESSING VIDEO -> NPY

Mục tiêu:
- Chọn một số video từ TEST 518.
- Lấy video MP4 gốc từ Celeb-DF v2.
- Chạy lại chính logic preprocessing dùng cho NPY.
- So sánh sequence mới với NPY đã dùng cho Test.

Pipeline được kiểm tra:
    np.linspace(..., 15)
    BGR -> RGB
    PIL
    MTCNN(
        image_size=224,
        margin=20,
        keep_all=False,
        select_largest=True,
        post_process=False
    )
    face crop 224x224
    fallback bằng face hợp lệ gần nhất
    padding bằng face cuối
    uint8 (15,224,224,3)

Chỉ kiểm tra preprocessing, KHÔNG chạy model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from inference_pipeline_finetune import (
    DATASET_ROOT,
    PROCESSED_ROOT,
    SEQUENCE_LENGTH,
    IMAGE_SIZE,
    VideoFacePreprocessor,
)


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(r"D:\Project")

TEST_PREDICTIONS = (
    PROJECT_DIR
    / "results_celebdf_finetune"
    / "finetune_test_predictions.csv"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results_celebdf_finetune"
    / "preprocess_consistency"
)

REPORT_FILE = (
    RESULTS_DIR
    / "preprocess_consistency_report.csv"
)

MAX_REAL = 3
MAX_FAKE = 3


# ============================================================
# MAP NPY -> ORIGINAL MP4
# ============================================================

def map_npy_to_mp4(npy_path: Path) -> Path:
    """
    Ví dụ:
        Processed_Data_CelebDF/Real/Celeb-real/id42_0002.npy
        -> Celeb-DF-v2/Celeb-real/id42_0002.mp4

        Processed_Data_CelebDF/Fake/Celeb-synthesis/id1_id2_0001.npy
        -> Celeb-DF-v2/Celeb-synthesis/id1_id2_0001.mp4
    """

    try:
        relative = npy_path.relative_to(
            PROCESSED_ROOT
        )
    except ValueError as exc:
        raise ValueError(
            f"NPY không thuộc PROCESSED_ROOT:\n{npy_path}\n"
            f"Root: {PROCESSED_ROOT}"
        ) from exc

    parts = relative.parts

    if len(parts) < 3:
        raise ValueError(
            f"Cấu trúc NPY không đúng: {relative}"
        )

    # parts:
    #   Real / Celeb-real / filename.npy
    #   Fake / Celeb-synthesis / filename.npy
    source_folder = parts[1]
    filename = Path(parts[-1]).with_suffix(".mp4").name

    mp4_path = (
        DATASET_ROOT
        / source_folder
        / filename
    )

    return mp4_path


# ============================================================
# SELECT TEST FILES
# ============================================================

def load_selected_predictions() -> pd.DataFrame:
    """Chọn một số Real/Fake đại diện từ Test 518."""

    if not TEST_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"Không tìm thấy:\n{TEST_PREDICTIONS}"
        )

    df = pd.read_csv(
        TEST_PREDICTIONS
    )

    required = {
        "npy_path",
        "label",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Thiếu cột: {sorted(missing)}"
        )

    real_df = (
        df[df["label"] == 0]
        .head(MAX_REAL)
        .copy()
    )

    fake_df = (
        df[df["label"] == 1]
        .head(MAX_FAKE)
        .copy()
    )

    selected = pd.concat(
        [real_df, fake_df],
        ignore_index=True,
    )

    if len(selected) != MAX_REAL + MAX_FAKE:
        raise RuntimeError(
            "Không chọn đủ mẫu Test để kiểm tra."
        )

    return selected


# ============================================================
# COMPARE
# ============================================================

def compare_arrays(
    generated: np.ndarray,
    stored: np.ndarray,
) -> dict:
    """Tính các sai khác giữa 2 sequence uint8."""

    if generated.shape != stored.shape:
        return {
            "same_shape": False,
            "exact_equal": False,
            "max_abs_diff": np.nan,
            "mean_abs_diff": np.nan,
            "different_pixel_ratio": np.nan,
        }

    diff = np.abs(
        generated.astype(np.int16)
        - stored.astype(np.int16)
    )

    return {
        "same_shape": True,
        "exact_equal": bool(
            np.array_equal(
                generated,
                stored,
            )
        ),
        "max_abs_diff": float(
            diff.max()
        ),
        "mean_abs_diff": float(
            diff.mean()
        ),
        "different_pixel_ratio": float(
            np.mean(diff != 0)
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    try:
        print("=" * 80)
        print("G8.1 - PREPROCESSING CONSISTENCY CHECK")
        print("=" * 80)

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"[INFO] Dataset root    : {DATASET_ROOT}"
        )
        print(
            f"[INFO] Processed root  : {PROCESSED_ROOT}"
        )

        selected = load_selected_predictions()

        # MTCNN được khởi tạo một lần cho toàn bộ kiểm tra.
        preprocessor = VideoFacePreprocessor()

        results = []

        for index, row in selected.iterrows():
            npy_path = Path(
                str(row["npy_path"])
            )

            label = int(
                row["label"]
            )

            class_name = (
                "Real" if label == 0 else "Fake"
            )

            print("\n" + "-" * 80)
            print(
                f"[{index + 1}/{len(selected)}] {class_name}"
            )
            print(
                f"NPY: {npy_path}"
            )

            if not npy_path.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy NPY:\n{npy_path}"
                )

            mp4_path = map_npy_to_mp4(
                npy_path
            )

            print(
                f"MP4: {mp4_path}"
            )

            if not mp4_path.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy MP4 gốc:\n{mp4_path}"
                )

            stored = np.load(
                npy_path,
                allow_pickle=False,
            )

            generated, metadata = (
                preprocessor.process_video(
                    mp4_path
                )
            )

            comparison = compare_arrays(
                generated,
                stored,
            )

            print(
                f"Shape stored    : {stored.shape}"
            )
            print(
                f"Shape generated : {generated.shape}"
            )
            print(
                f"Exact equal     : {comparison['exact_equal']}"
            )
            print(
                f"Max abs diff    : {comparison['max_abs_diff']}"
            )
            print(
                f"Mean abs diff   : {comparison['mean_abs_diff']:.6f}"
            )
            print(
                f"Diff pixel ratio: {comparison['different_pixel_ratio']:.6%}"
            )
            print(
                f"Detected faces  : {metadata['detected_faces']}"
            )
            print(
                f"Fallback faces  : {metadata['fallback_faces']}"
            )
            print(
                f"Read failures   : {metadata['read_failures']}"
            )

            result_row = {
                "npy_path": str(npy_path),
                "mp4_path": str(mp4_path),
                "label": label,
                "class": class_name,
                "same_shape": comparison["same_shape"],
                "exact_equal": comparison["exact_equal"],
                "max_abs_diff": comparison["max_abs_diff"],
                "mean_abs_diff": comparison["mean_abs_diff"],
                "different_pixel_ratio": comparison["different_pixel_ratio"],
                "detected_faces": metadata["detected_faces"],
                "fallback_faces": metadata["fallback_faces"],
                "read_failures": metadata["read_failures"],
            }

            results.append(result_row)

        result_df = pd.DataFrame(
            results
        )

        result_df.to_csv(
            REPORT_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        exact_count = int(
            result_df["exact_equal"].sum()
        )

        shape_count = int(
            result_df["same_shape"].sum()
        )

        print("\n" + "=" * 80)
        print("KẾT LUẬN G8.1")
        print("=" * 80)
        print(
            f"Samples checked : {len(result_df)}"
        )
        print(
            f"Shape matched   : {shape_count}/{len(result_df)}"
        )
        print(
            f"Exact matched   : {exact_count}/{len(result_df)}"
        )
        print(
            f"Report          : {REPORT_FILE}"
        )

        if exact_count == len(result_df):
            print(
                "[PASS] Preprocessing video -> NPY "
                "đồng bộ hoàn toàn trên các mẫu kiểm tra."
            )
        else:
            print(
                "[WARNING] Có sai khác giữa preprocessing "
                "tạo lại và NPY đã lưu."
            )
            print(
                "Chưa nên dùng pipeline raw-video cho Streamlit "
                "cho tới khi kiểm tra nguyên nhân sai khác."
            )

        print("=" * 80)

    except KeyboardInterrupt:
        print("\n[INFO] Người dùng dừng kiểm tra.")
        sys.exit(1)
    except Exception as exc:
        print("\n[ERROR] G8.1 consistency check thất bại.")
        print(f"[ERROR] {exc}")
        raise


if __name__ == "__main__":
    main()
