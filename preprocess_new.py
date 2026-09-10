import os
import csv

import cv2
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN


class VideoPreprocessor:
    """
    Pipeline tiền xử lý video cho bài toán phát hiện Deepfake.

    Quy trình:
        Video
        -> Uniform Frame Sampling
        -> Face Detection bằng MTCNN
        -> Face Crop
        -> Resize 224x224
        -> Sequence N frames
        -> Lưu .npy
    """

    def __init__(
        self,
        sequence_length=15,
        image_size=224,
        margin=20
    ):
        self.seq_len = sequence_length
        self.img_size = image_size

        # ---------------------------------------------------------
        # 1. Chọn thiết bị
        # ---------------------------------------------------------
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"[INFO] Thiết bị sử dụng: {self.device}")

        # ---------------------------------------------------------
        # 2. Khởi tạo MTCNN
        # ---------------------------------------------------------
        self.mtcnn = MTCNN(
            image_size=self.img_size,
            margin=margin,
            keep_all=False,
            post_process=False,
            device=self.device
        )

    # =============================================================
    # HÀM 1: TRÍCH XUẤT SEQUENCE TỪ MỘT VIDEO
    # =============================================================
    def process_video(self, video_path):
        """
        Xử lý một video và trả về sequence khuôn mặt.

        Returns:
            numpy.ndarray:
                Shape = (sequence_length, image_size, image_size, 3)

            None:
                Nếu xử lý thất bại.
        """

        cap = None

        try:
            # -----------------------------------------------------
            # Mở video
            # -----------------------------------------------------
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise ValueError(
                    f"Không thể mở video: {video_path}"
                )

            # -----------------------------------------------------
            # Lấy tổng số frame
            # -----------------------------------------------------
            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if total_frames <= 0:
                raise ValueError(
                    "Video không có frame hợp lệ."
                )

            # -----------------------------------------------------
            # Kiểm tra video đủ dài
            # -----------------------------------------------------
            if total_frames < self.seq_len:
                raise ValueError(
                    f"Video chỉ có {total_frames} frame, "
                    f"yêu cầu tối thiểu {self.seq_len} frame."
                )

            # -----------------------------------------------------
            # Uniform Temporal Sampling
            # -----------------------------------------------------
            frame_indices = np.linspace(
                0,
                total_frames - 1,
                self.seq_len,
                dtype=np.int32
            )

            face_sequence = []

            # Lưu khuôn mặt hợp lệ gần nhất
            last_valid_face = None

            # -----------------------------------------------------
            # Xử lý từng frame
            # -----------------------------------------------------
            for frame_idx in frame_indices:

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    int(frame_idx)
                )

                ret, frame = cap.read()

                # -------------------------------------------------
                # Không đọc được frame
                # -------------------------------------------------
                if not ret:
                    if last_valid_face is not None:
                        face_sequence.append(
                            last_valid_face.copy()
                        )
                    continue

                # -------------------------------------------------
                # BGR -> RGB
                # -------------------------------------------------
                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # -------------------------------------------------
                # OpenCV -> PIL
                # -------------------------------------------------
                image = Image.fromarray(frame_rgb)

                # -------------------------------------------------
                # Face Detection + Crop
                # -------------------------------------------------
                face = self.mtcnn(image)

                # -------------------------------------------------
                # Phát hiện được khuôn mặt
                # -------------------------------------------------
                if face is not None:

                    face_np = (
                        face
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                        .clip(0, 255)
                        .astype(np.uint8)
                    )

                    # Lưu frame hợp lệ gần nhất
                    last_valid_face = face_np

                    face_sequence.append(
                        face_np.copy()
                    )

                else:
                    # -------------------------------------------------
                    # Không phát hiện mặt:
                    # dùng face hợp lệ gần nhất
                    # -------------------------------------------------
                    if last_valid_face is not None:
                        face_sequence.append(
                            last_valid_face.copy()
                        )

            # -----------------------------------------------------
            # Nếu sequence vẫn thiếu -> padding
            # -----------------------------------------------------
            if len(face_sequence) > 0:

                while len(face_sequence) < self.seq_len:

                    face_sequence.append(
                        face_sequence[-1].copy()
                    )

            # -----------------------------------------------------
            # Kiểm tra kết quả
            # -----------------------------------------------------
            if len(face_sequence) != self.seq_len:

                return None

            sequence = np.asarray(
                face_sequence,
                dtype=np.uint8
            )

            # -----------------------------------------------------
            # Kiểm tra shape
            # -----------------------------------------------------
            expected_shape = (
                self.seq_len,
                self.img_size,
                self.img_size,
                3
            )

            if sequence.shape != expected_shape:

                raise ValueError(
                    f"Shape không đúng: {sequence.shape}, "
                    f"mong đợi {expected_shape}"
                )

            return sequence

        except Exception as e:

            print(
                f"[ERROR] Không thể xử lý video "
                f"{video_path}: {e}"
            )

            return None

        finally:

            # -----------------------------------------------------
            # Đảm bảo video luôn được giải phóng
            # -----------------------------------------------------
            if cap is not None:
                cap.release()

    # =============================================================
    # HÀM 2: XỬ LÝ TOÀN BỘ VIDEO TRONG MỘT THƯ MỤC
    # =============================================================
    def process_dataset(
        self,
        input_dir,
        output_dir,
        label_name
    ):
        """
        Xử lý toàn bộ video trong một thư mục.

        label_name:
            Real / Deepfakes / Face2Face / ...
        """

        if not os.path.exists(input_dir):

            print(
                f"[WARNING] Không tồn tại: {input_dir}"
            )
            return []

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        # ---------------------------------------------------------
        # Lọc video và sort để đảm bảo thứ tự tái lập
        # ---------------------------------------------------------
        video_files = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov"))
        ])

        print(
            f"[INFO] {label_name}: "
            f"{len(video_files)} video"
        )

        successful_files = []
        failed_files = []

        # ---------------------------------------------------------
        # Xử lý từng video
        # ---------------------------------------------------------
        for video_name in tqdm(
            video_files,
            desc=f"Processing {label_name}"
        ):

            video_path = os.path.join(
                input_dir,
                video_name
            )

            # -----------------------------------------------------
            # Tạo tên .npy an toàn
            # -----------------------------------------------------
            base_name = os.path.splitext(
                video_name
            )[0]

            output_path = os.path.join(
                output_dir,
                base_name + ".npy"
            )

            # -----------------------------------------------------
            # Đã xử lý -> bỏ qua
            # -----------------------------------------------------
            if os.path.exists(output_path):

                successful_files.append(
                    video_name
                )

                continue

            # -----------------------------------------------------
            # Xử lý video
            # -----------------------------------------------------
            sequence = self.process_video(
                video_path
            )

            # -----------------------------------------------------
            # Thành công
            # -----------------------------------------------------
            if sequence is not None:

                np.save(
                    output_path,
                    sequence
                )

                successful_files.append(
                    video_name
                )

            else:

                failed_files.append(
                    video_name
                )

        # ---------------------------------------------------------
        # Thống kê
        # ---------------------------------------------------------
        print(
            f"[INFO] {label_name} hoàn thành:"
        )

        print(
            f"       Thành công: "
            f"{len(successful_files)}/{len(video_files)}"
        )

        print(
            f"       Thất bại: "
            f"{len(failed_files)}"
        )

        return successful_files, failed_files


# =============================================================
# HÀM 3: TẠO MANIFEST
# =============================================================
def create_manifest(
    processed_root,
    manifest_path
):
    """
    Tạo file CSV mô tả toàn bộ dữ liệu đã preprocessing.

    Manifest giúp kiểm tra và quản lý dataset
    trước khi chia Train / Validation / Test.
    """

    rows = []

    # ---------------------------------------------------------
    # REAL
    # ---------------------------------------------------------
    real_dir = os.path.join(
        processed_root,
        "Real"
    )

    if os.path.exists(real_dir):

        for file_name in sorted(
            os.listdir(real_dir)
        ):

            if file_name.endswith(".npy"):

                rows.append([
                    file_name,
                    "Real",
                    0,
                    os.path.join(
                        "Real",
                        file_name
                    )
                ])

    # ---------------------------------------------------------
    # FAKE
    # ---------------------------------------------------------
    fake_root = os.path.join(
        processed_root,
        "Fake"
    )

    if os.path.exists(fake_root):

        for method in sorted(
            os.listdir(fake_root)
        ):

            method_dir = os.path.join(
                fake_root,
                method
            )

            if not os.path.isdir(method_dir):
                continue

            for file_name in sorted(
                os.listdir(method_dir)
            ):

                if file_name.endswith(".npy"):

                    rows.append([
                        file_name,
                        method,
                        1,
                        os.path.join(
                            "Fake",
                            method,
                            file_name
                        )
                    ])

    # ---------------------------------------------------------
    # Ghi CSV
    # ---------------------------------------------------------
    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "video_name",
            "method",
            "label",
            "relative_path"
        ])

        writer.writerows(rows)

    print(
        f"[INFO] Manifest đã tạo: "
        f"{manifest_path}"
    )

    print(
        f"[INFO] Tổng số sequence: "
        f"{len(rows)}"
    )


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":

    # ---------------------------------------------------------
    # Cấu hình
    # ---------------------------------------------------------
    preprocessor = VideoPreprocessor(
        sequence_length=15,
        image_size=224,
        margin=20
    )

    base_data_dir = r"D:\Project\FaceForensicsData"
    output_base_dir = r"D:\Project\Processed_Data_v2"
    # ---------------------------------------------------------
    # 1. REAL
    # ---------------------------------------------------------
    real_input_dir = os.path.join(
        base_data_dir,
        "original_sequences",
        "youtube",
        "c23",
        "videos"
    )

    real_output_dir = os.path.join(
        output_base_dir,
        "Real"
    )

    print(
        "\n================ REAL ================\n"
    )

    preprocessor.process_dataset(
        real_input_dir,
        real_output_dir,
        "Real"
    )

    # ---------------------------------------------------------
    # 2. FAKE
    # ---------------------------------------------------------
    fake_methods = [
        "Deepfakes",
        "Face2Face",
        "FaceShifter",
        "FaceSwap",
        "NeuralTextures"
    ]

    print(
        "\n=============== FAKE ================\n"
    )

    for method in fake_methods:

        fake_input_dir = os.path.join(
            base_data_dir,
            "manipulated_sequences",
            method,
            "c23",
            "videos"
        )

        fake_output_dir = os.path.join(
            output_base_dir,
            "Fake",
            method
        )

        print(
            f"\n---------- {method} ----------"
        )

        preprocessor.process_dataset(
            fake_input_dir,
            fake_output_dir,
            method
        )

    # ---------------------------------------------------------
    # 3. Tạo manifest
    # ---------------------------------------------------------
    manifest_path = os.path.join(
        output_base_dir,
        "dataset_manifest.csv"
    )

    create_manifest(
        output_base_dir,
        manifest_path
    )

    print(
        "\n[INFO] HOÀN TẤT GIAI ĐOẠN 1"
    )