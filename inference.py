import os
import cv2
import numpy as np
import torch

from PIL import Image
from facenet_pytorch import MTCNN

from model import SpatialTemporalXceptionBiLSTM


# ============================================================
# 1. CẤU HÌNH
# ============================================================

CHECKPOINT_PATH = (
    r"D:\Project\checkpoints\best_model.pth"
)

SEQUENCE_LENGTH = 15
IMAGE_SIZE = 224

LSTM_HIDDEN_SIZE = 256
LSTM_LAYERS = 2

# Ngưỡng hiện tại dùng cho hệ thống demo
# Không dùng threshold 0.90 vì thực nghiệm trước cho thấy
# threshold 0.50 phù hợp hơn cho mục tiêu phát hiện Fake.
THRESHOLD = 0.50


# ============================================================
# 2. DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            f"[INFO] Device: {device}"
        )

        print(
            f"[INFO] GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        device = torch.device("cpu")

        print(
            "[WARNING] CUDA không khả dụng."
        )

    return device


# ============================================================
# 3. LOAD MODEL
# ============================================================

def load_model(device):

    if not os.path.exists(
        CHECKPOINT_PATH
    ):

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

    return model


# ============================================================
# 4. KHỞI TẠO MTCNN
# ============================================================

def create_mtcnn(device):

    print(
        "\n========== KHỞI TẠO MTCNN =========="
    )

    mtcnn = MTCNN(
        image_size=IMAGE_SIZE,
        margin=20,
        keep_all=False,
        select_largest=True,
        post_process=False,
        device=device
    )

    print(
        "[INFO] MTCNN đã khởi tạo."
    )

    return mtcnn


# ============================================================
# 5. LẤY 15 FRAME ĐỒNG ĐỀU
# ============================================================

def sample_frame_indices(
    total_frames,
    sequence_length
):

    if total_frames <= 0:

        raise ValueError(
            "Video không chứa frame."
        )

    indices = np.linspace(
        0,
        total_frames - 1,
        sequence_length
    ).astype(int)

    return indices


# ============================================================
# 6. ĐỌC VIDEO + DETECT FACE
# ============================================================

def extract_face_sequence(
    video_path,
    mtcnn
):

    if not os.path.exists(
        video_path
    ):

        raise FileNotFoundError(
            f"Không tìm thấy video:\n"
            f"{video_path}"
        )

    print(
        "\n========== XỬ LÝ VIDEO =========="
    )

    print(
        f"[INFO] Video: {video_path}"
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Không thể mở video."
        )

    try:

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if total_frames <= 0:

            raise RuntimeError(
                "Không đọc được số frame của video."
            )

        frame_indices = sample_frame_indices(
            total_frames,
            SEQUENCE_LENGTH
        )

        print(
            f"[INFO] Total frames: "
            f"{total_frames}"
        )

        print(
            f"[INFO] Sample indices: "
            f"{frame_indices.tolist()}"
        )

        face_sequence = []

        last_valid_face = None

        detected_count = 0

        for frame_index in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index)
            )

            success, frame = cap.read()

            if not success:

                print(
                    f"[WARNING] Không đọc được "
                    f"frame {frame_index}"
                )

                if last_valid_face is not None:

                    face = last_valid_face.copy()

                    face_sequence.append(face)

                continue

            # BGR -> RGB
            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # RGB -> PIL
            pil_image = Image.fromarray(
                frame_rgb
            )

            # MTCNN face detection + crop
            face = mtcnn(
                pil_image
            )

            if face is not None:

                face = face.permute(
                    1,
                    2,
                    0
                ).cpu().numpy()

                face = np.clip(
                    face,
                    0,
                    255
                ).astype(
                    np.uint8
                )

                last_valid_face = face.copy()

                face_sequence.append(
                    face
                )

                detected_count += 1

            else:

                print(
                    f"[WARNING] Không phát hiện "
                    f"khuôn mặt tại frame "
                    f"{frame_index}"
                )

                if last_valid_face is not None:

                    face_sequence.append(
                        last_valid_face.copy()
                    )

        # ----------------------------------------------------
        # Nếu không có frame nào phát hiện được mặt
        # ----------------------------------------------------

        if len(face_sequence) == 0:

            raise RuntimeError(
                "Không phát hiện được khuôn mặt "
                "trong video."
            )

        # ----------------------------------------------------
        # Bù đủ 15 frame
        # ----------------------------------------------------

        while len(face_sequence) < SEQUENCE_LENGTH:

            face_sequence.append(
                face_sequence[-1].copy()
            )

        face_sequence = face_sequence[
            :SEQUENCE_LENGTH
        ]

        sequence = np.stack(
            face_sequence,
            axis=0
        )

        print(
            f"[INFO] Face detected: "
            f"{detected_count}/{SEQUENCE_LENGTH}"
        )

        print(
            f"[INFO] Sequence shape: "
            f"{sequence.shape}"
        )

        return sequence

    finally:

        cap.release()


# ============================================================
# 7. PREPARE TENSOR
# ============================================================

def prepare_tensor(sequence):

    if sequence.shape != (
        SEQUENCE_LENGTH,
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    ):

        raise ValueError(
            f"Sequence shape không đúng: "
            f"{sequence.shape}"
        )

    # uint8 -> float32 [0, 1]
    tensor = torch.from_numpy(
        sequence.copy()
    ).float() / 255.0

    # NHWC -> NCHW
    tensor = tensor.permute(
        0, 3, 1, 2
    )

    # ImageNet normalization
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=torch.float32
    ).view(
        1, 3, 1, 1
    )

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=torch.float32
    ).view(
        1, 3, 1, 1
    )

    tensor = (
        tensor - mean
    ) / std

    # Thêm batch dimension
    tensor = tensor.unsqueeze(
        0
    )

    return tensor


# ============================================================
# 8. PREDICT
# ============================================================

@torch.no_grad()
def predict_video(
    model,
    mtcnn,
    video_path,
    device
):

    sequence = extract_face_sequence(
        video_path,
        mtcnn
    )

    tensor = prepare_tensor(
        sequence
    )

    tensor = tensor.to(
        device
    )

    logits = model(
        tensor
    )

    logit = logits.squeeze().item()

    probability_fake = float(
        torch.sigmoid(
            torch.tensor(logit)
        ).item()
    )

    probability_real = (
        1.0 - probability_fake
    )

    if probability_fake >= THRESHOLD:

        prediction = "FAKE"

        confidence = probability_fake

    else:

        prediction = "REAL"

        confidence = probability_real

    result = {
        "prediction": prediction,

        "probability_fake":
            probability_fake,

        "probability_real":
            probability_real,

        "confidence":
            confidence,

        "threshold":
            THRESHOLD,

        "sequence": sequence
    }

    return result


# ============================================================
# 9. MAIN TEST
# ============================================================

def main():

    print("\n")

    print("=" * 65)

    print(
        "       DEEPFAKE DETECTION"
    )

    print(
        "       VIDEO INFERENCE TEST"
    )

    print("=" * 65)

    device = get_device()

    model = load_model(
        device
    )

    mtcnn = create_mtcnn(
        device
    )

    # --------------------------------------------------------
    # NHẬP ĐƯỜNG DẪN VIDEO
    # --------------------------------------------------------

    video_path = input(
        "\nNhập đường dẫn tới video cần kiểm tra: "
    ).strip()

    if not video_path:

        raise ValueError(
            "Bạn chưa nhập đường dẫn video."
        )

    result = predict_video(
        model=model,
        mtcnn=mtcnn,
        video_path=video_path,
        device=device
    )

    # --------------------------------------------------------
    # KẾT QUẢ
    # --------------------------------------------------------

    print("\n")

    print("=" * 65)

    print(
        "               KẾT QUẢ"
    )

    print("=" * 65)

    print(
        f"Dự đoán: "
        f"{result['prediction']}"
    )

    print(
        f"Xác suất Real: "
        f"{result['probability_real'] * 100:.2f}%"
    )

    print(
        f"Xác suất Fake: "
        f"{result['probability_fake'] * 100:.2f}%"
    )

    print(
        f"Confidence: "
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        f"Threshold: "
        f"{result['threshold']:.2f}"
    )

    print("\n[OK] Inference hoàn tất.")


if __name__ == "__main__":
    main()