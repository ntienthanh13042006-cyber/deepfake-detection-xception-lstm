import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch

from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN

from model import SpatialTemporalXceptionBiLSTM


# ============================================================
# 1. CẤU HÌNH HỆ THỐNG
# ============================================================

CHECKPOINT_PATH = (
    r"D:\Project\checkpoints\best_model.pth"
)

SEQUENCE_LENGTH = 15
IMAGE_SIZE = 224

LSTM_HIDDEN_SIZE = 256
LSTM_LAYERS = 2

# Threshold được khóa theo thực nghiệm hiện tại
THRESHOLD = 0.50


# ============================================================
# 2. CẤU HÌNH TRANG STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Deepfake Video Detection",
    page_icon="🎥",
    layout="wide"
)


# ============================================================
# 3. DEVICE
# ============================================================

@st.cache_resource
def get_device():

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    device = get_device()

    if not os.path.exists(CHECKPOINT_PATH):

        raise FileNotFoundError(
            f"Không tìm thấy checkpoint:\n"
            f"{CHECKPOINT_PATH}"
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

    return model, checkpoint


# ============================================================
# 5. LOAD MTCNN
# ============================================================

@st.cache_resource
def load_mtcnn():

    device = get_device()

    mtcnn = MTCNN(
        image_size=IMAGE_SIZE,
        margin=20,
        keep_all=False,
        select_largest=True,
        post_process=False,
        device=device
    )

    return mtcnn


# ============================================================
# 6. LẤY THÔNG TIN VIDEO
# ============================================================

def get_video_info(video_path):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Không thể mở video để đọc thông tin."
        )

    try:

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = float(
            cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        if fps <= 0:

            fps = 0.0

        if fps > 0:

            duration = (
                total_frames / fps
            )

        else:

            duration = 0.0

        return {
            "total_frames": total_frames,
            "fps": fps,
            "duration": duration
        }

    finally:

        cap.release()


# ============================================================
# 7. SAMPLE 15 FRAME
# ============================================================

def sample_frame_indices(
    total_frames
):

    if total_frames <= 0:

        raise ValueError(
            "Video không có frame."
        )

    indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH
    ).astype(int)

    return indices


# ============================================================
# 8. EXTRACT FACE + VISUALIZATION
# ============================================================

def extract_face_sequence(
    video_path,
    mtcnn
):

    if not os.path.exists(video_path):

        raise FileNotFoundError(
            f"Không tìm thấy video:\n"
            f"{video_path}"
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
            total_frames
        )

        face_sequence = []

        last_valid_face = None

        detected_count = 0

        visualization_items = []

        for frame_index in frame_indices:

            # ------------------------------------------------
            # Đưa con trỏ video đến frame cần lấy
            # ------------------------------------------------

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index)
            )

            success, frame = cap.read()

            if not success:

                if last_valid_face is not None:

                    face_sequence.append(
                        last_valid_face.copy()
                    )

                continue

            # ------------------------------------------------
            # BGR -> RGB
            # ------------------------------------------------

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            pil_image = Image.fromarray(
                frame_rgb
            )

            # ------------------------------------------------
            # Detect + crop face
            # ------------------------------------------------

            face = mtcnn(
                pil_image
            )

            if face is not None:

                # Tensor C,H,W -> H,W,C
                face = (
                    face
                    .permute(
                        1,
                        2,
                        0
                    )
                    .cpu()
                    .numpy()
                )

                face = np.clip(
                    face,
                    0,
                    255
                ).astype(
                    np.uint8
                )

                last_valid_face = (
                    face.copy()
                )

                face_sequence.append(
                    face
                )

                detected_count += 1

                # ------------------------------------------------
                # Tạo visualization cho tối đa 5 frame đầu tiên
                # ------------------------------------------------

                if len(
                    visualization_items
                ) < 5:

                    annotated_image = (
                        pil_image.copy()
                    )

                    draw = ImageDraw.Draw(
                        annotated_image
                    )

                    # Detect bounding box
                    # Chỉ gọi thêm detect cho frame
                    # được dùng để trực quan hóa
                    try:

                        boxes, probabilities = (
                            mtcnn.detect(
                                pil_image
                            )
                        )

                        if boxes is not None:

                            # Chọn khuôn mặt lớn nhất
                            best_box = None
                            best_area = -1

                            for box in boxes:

                                x1, y1, x2, y2 = (
                                    box
                                )

                                area = max(
                                    0,
                                    x2 - x1
                                ) * max(
                                    0,
                                    y2 - y1
                                )

                                if area > best_area:

                                    best_area = area
                                    best_box = box

                            if best_box is not None:

                                x1, y1, x2, y2 = (
                                    best_box
                                )

                                x1 = int(
                                    max(
                                        0,
                                        x1
                                    )
                                )

                                y1 = int(
                                    max(
                                        0,
                                        y1
                                    )
                                )

                                x2 = int(
                                    min(
                                        pil_image.width,
                                        x2
                                    )
                                )

                                y2 = int(
                                    min(
                                        pil_image.height,
                                        y2
                                    )
                                )

                                draw.rectangle(
                                    [
                                        x1,
                                        y1,
                                        x2,
                                        y2
                                    ],
                                    outline=(
                                        255,
                                        0,
                                        0
                                    ),
                                    width=4
                                )

                                draw.text(
                                    (
                                        x1,
                                        max(
                                            0,
                                            y1 - 20
                                        )
                                    ),
                                    "FACE"
                                )

                    except Exception:

                        # Nếu visualization box lỗi
                        # vẫn giữ frame gốc
                        pass

                    visualization_items.append(
                        {
                            "frame_index":
                                int(frame_index),

                            "original":
                                np.array(
                                    annotated_image
                                ),

                            "face":
                                face.copy()
                        }
                    )

            else:

                # ------------------------------------------------
                # Nếu không detect được face
                # dùng face hợp lệ gần nhất
                # ------------------------------------------------

                if last_valid_face is not None:

                    face_sequence.append(
                        last_valid_face.copy()
                    )

        # ========================================================
        # Kiểm tra
        # ========================================================

        if len(face_sequence) == 0:

            raise RuntimeError(
                "Không phát hiện được khuôn mặt "
                "trong video."
            )

        # ========================================================
        # Padding đủ 15 frame
        # ========================================================

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

        return (
            sequence,
            detected_count,
            visualization_items,
            total_frames,
            frame_indices
        )

    finally:

        cap.release()


# ============================================================
# 9. PREPARE TENSOR
# ============================================================

def prepare_tensor(sequence):

    expected_shape = (
        SEQUENCE_LENGTH,
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    )

    if sequence.shape != expected_shape:

        raise ValueError(
            f"Shape không đúng: "
            f"{sequence.shape}; "
            f"mong đợi {expected_shape}"
        )

    # --------------------------------------------------------
    # uint8 -> float32 [0,1]
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        sequence.copy()
    ).float() / 255.0

    # --------------------------------------------------------
    # NHWC -> NCHW
    # --------------------------------------------------------

    tensor = tensor.permute(
        0,
        3,
        1,
        2
    )

    # --------------------------------------------------------
    # ImageNet normalization
    # --------------------------------------------------------

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=torch.float32
    ).view(
        1,
        3,
        1,
        1
    )

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=torch.float32
    ).view(
        1,
        3,
        1,
        1
    )

    tensor = (
        tensor - mean
    ) / std

    # --------------------------------------------------------
    # Thêm batch dimension
    # --------------------------------------------------------

    tensor = tensor.unsqueeze(
        0
    )

    return tensor


# ============================================================
# 10. PREDICT VIDEO
# ============================================================

@torch.no_grad()
def predict_video(
    model,
    mtcnn,
    video_path
):

    device = get_device()

    (
        sequence,
        detected_count,
        visualization_items,
        total_frames,
        frame_indices
    ) = extract_face_sequence(
        video_path,
        mtcnn
    )

    tensor = prepare_tensor(
        sequence
    )

    tensor = tensor.to(
        device
    )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    logits = model(
        tensor
    )

    logit = logits.squeeze().item()

    # --------------------------------------------------------
    # Sigmoid
    # --------------------------------------------------------

    fake_probability = float(
        torch.sigmoid(
            torch.tensor(
                logit
            )
        ).item()
    )

    real_probability = (
        1.0 - fake_probability
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if fake_probability >= THRESHOLD:

        prediction = "FAKE"

        confidence = fake_probability

    else:

        prediction = "REAL"

        confidence = real_probability

    return {
        "prediction":
            prediction,

        "fake_probability":
            fake_probability,

        "real_probability":
            real_probability,

        "confidence":
            confidence,

        "threshold":
            THRESHOLD,

        "detected_count":
            detected_count,

        "total_frames":
            total_frames,

        "frame_indices":
            frame_indices,

        "visualization_items":
            visualization_items
    }


# ============================================================
# 11. HEADER
# ============================================================

st.title(
    "🎥 Hệ thống phát hiện video Deepfake"
)

st.write(
    "Phát hiện video giả mạo dựa trên "
    "đặc trưng khuôn mặt sử dụng "
    "Xception + Bi-LSTM."
)

st.divider()


# ============================================================
# 12. LOAD RESOURCES
# ============================================================

try:

    model, checkpoint = load_model()

    mtcnn = load_mtcnn()

    device = get_device()

except Exception as error:

    st.error(
        "Không thể khởi tạo hệ thống."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# 13. SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Thông tin mô hình"
    )

    st.write(
        f"**Device:** `{device}`"
    )

    if torch.cuda.is_available():

        st.write(
            f"**GPU:** "
            f"`{torch.cuda.get_device_name(0)}`"
        )

    checkpoint_epoch = checkpoint.get(
        "epoch",
        "N/A"
    )

    checkpoint_auc = checkpoint.get(
        "val_auc",
        None
    )

    st.write(
        f"**Checkpoint epoch:** "
        f"`{checkpoint_epoch}`"
    )

    if checkpoint_auc is not None:

        st.write(
            f"**Validation ROC-AUC:** "
            f"`{checkpoint_auc:.4f}`"
        )

    st.write(
        f"**Sequence:** "
        f"`{SEQUENCE_LENGTH} frames`"
    )

    st.write(
        f"**Face size:** "
        f"`{IMAGE_SIZE} × {IMAGE_SIZE}`"
    )

    st.write(
        f"**Threshold:** "
        f"`{THRESHOLD:.2f}`"
    )

    st.divider()

    st.caption(
        "Model: Xception + Bi-LSTM"
    )

    st.caption(
        "Xception feature: 2048"
    )

    st.caption(
        "Bi-LSTM hidden size: 256"
    )

    st.caption(
        "Bi-LSTM layers: 2"
    )


# ============================================================
# 14. UPLOAD VIDEO
# ============================================================

uploaded_file = st.file_uploader(
    "📁 Chọn video cần kiểm tra",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv"
    ]
)


# ============================================================
# 15. VIDEO PREVIEW
# ============================================================

if uploaded_file is not None:

    st.subheader(
        "🎬 Video đầu vào"
    )

    try:

        uploaded_bytes = (
            uploaded_file.getvalue()
        )

        st.video(
            uploaded_bytes
        )

    except Exception as error:

        st.warning(
            f"Không thể hiển thị preview "
            f"video: {error}"
        )

    # --------------------------------------------------------
    # Nút phân tích
    # --------------------------------------------------------

    analyze_button = st.button(
        "🔍 Phân tích video",
        type="primary",
        use_container_width=True
    )

    # ========================================================
    # 16. ANALYZE
    # ========================================================

    if analyze_button:

        temporary_path = None

        try:

            # ------------------------------------------------
            # Lấy extension
            # ------------------------------------------------

            suffix = os.path.splitext(
                uploaded_file.name
            )[1]

            if not suffix:

                suffix = ".mp4"

            # ------------------------------------------------
            # Tạo file tạm
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(
                    uploaded_bytes
                )

                temporary_path = (
                    temp_file.name
                )

            # ------------------------------------------------
            # Video information
            # ------------------------------------------------

            video_info = get_video_info(
                temporary_path
            )

            # ------------------------------------------------
            # Phân tích
            # ------------------------------------------------

            with st.spinner(
                "Đang phát hiện khuôn mặt "
                "và phân tích video..."
            ):

                result = predict_video(
                    model=model,
                    mtcnn=mtcnn,
                    video_path=temporary_path
                )

            # =================================================
            # THÔNG BÁO HOÀN TẤT
            # =================================================

            st.success(
                "✅ Phân tích video hoàn tất!"
            )

            st.divider()

            # =================================================
            # KẾT QUẢ
            # =================================================

            st.subheader(
                "🔎 Kết quả phân tích"
            )

            prediction = (
                result["prediction"]
            )

            fake_probability = (
                result["fake_probability"]
            )

            real_probability = (
                result["real_probability"]
            )

            confidence = (
                result["confidence"]
            )

            # ------------------------------------------------
            # Kết luận lớn
            # ------------------------------------------------

            if prediction == "FAKE":

                st.error(
                    "🚨 KẾT QUẢ: VIDEO CÓ KHẢ NĂNG GIẢ MẠO (FAKE)"
                )

            else:

                st.success(
                    "✅ KẾT QUẢ: VIDEO CÓ KHẢ NĂNG THẬT (REAL)"
                )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Xác suất REAL",
                    f"{real_probability * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "Xác suất FAKE",
                    f"{fake_probability * 100:.2f}%"
                )

            with col3:

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

            # =================================================
            # PROBABILITY BAR
            # =================================================

            st.write(
                "### 📈 Xác suất dự đoán"
            )

            st.progress(
                fake_probability,
                text=(
                    f"Fake probability: "
                    f"{fake_probability * 100:.2f}%"
                )
            )

            st.progress(
                real_probability,
                text=(
                    f"Real probability: "
                    f"{real_probability * 100:.2f}%"
                )
            )

            st.caption(
                f"Ngưỡng phân loại hiện tại: "
                f"{THRESHOLD:.2f}"
            )

            # =================================================
            # VIDEO INFORMATION
            # =================================================

            st.divider()

            st.subheader(
                "📊 Thông tin video"
            )

            info_col1, info_col2, info_col3, info_col4 = (
                st.columns(4)
            )

            with info_col1:

                st.metric(
                    "Tổng số frame",
                    f"{video_info['total_frames']}"
                )

            with info_col2:

                if video_info["fps"] > 0:

                    st.metric(
                        "FPS",
                        f"{video_info['fps']:.2f}"
                    )

                else:

                    st.metric(
                        "FPS",
                        "N/A"
                    )

            with info_col3:

                if video_info["duration"] > 0:

                    st.metric(
                        "Thời lượng",
                        f"{video_info['duration']:.2f}s"
                    )

                else:

                    st.metric(
                        "Thời lượng",
                        "N/A"
                    )

            with info_col4:

                st.metric(
                    "Frame sử dụng",
                    f"{SEQUENCE_LENGTH}"
                )

            # =================================================
            # FACE DETECTION INFORMATION
            # =================================================

            st.divider()

            st.subheader(
                "👤 Thông tin phát hiện khuôn mặt"
            )

            detected_count = (
                result["detected_count"]
            )

            detected_ratio = (
                detected_count /
                SEQUENCE_LENGTH
            )

            face_col1, face_col2 = st.columns(2)

            with face_col1:

                st.metric(
                    "Frame phát hiện được mặt",
                    f"{detected_count}/{SEQUENCE_LENGTH}"
                )

            with face_col2:

                st.metric(
                    "Tỷ lệ phát hiện",
                    f"{detected_ratio * 100:.2f}%"
                )

            # ------------------------------------------------
            # Quality warning
            # ------------------------------------------------

            if detected_ratio < 0.50:

                st.error(
                    "⚠️ Chỉ phát hiện được khuôn mặt "
                    "ở dưới 50% frame. "
                    "Kết quả có thể kém tin cậy."
                )

            elif detected_ratio < 0.80:

                st.warning(
                    "⚠️ Một số frame không phát hiện "
                    "được khuôn mặt. "
                    "Hệ thống sử dụng khuôn mặt hợp lệ "
                    "gần nhất để bổ sung."
                )

            else:

                st.info(
                    "✓ Chất lượng phát hiện khuôn mặt tốt."
                )

            # =================================================
            # VISUALIZATION
            # =================================================

            st.divider()

            st.subheader(
                "👁️ Trực quan hóa khuôn mặt"
            )

            st.write(
                "Các frame dưới đây minh họa "
                "khuôn mặt được phát hiện và vùng "
                "khuôn mặt được trích xuất."
            )

            visualization_items = (
                result[
                    "visualization_items"
                ]
            )

            if len(
                visualization_items
            ) > 0:

                st.write(
                    "### Frame gốc + Bounding Box"
                )

                original_columns = st.columns(
                    len(
                        visualization_items
                    )
                )

                for column, item in zip(
                    original_columns,
                    visualization_items
                ):

                    with column:

                        st.image(
                            item["original"],
                            caption=(
                                f"Frame "
                                f"{item['frame_index']}"
                            ),
                            use_container_width=True
                        )

                st.write(
                    "### Face Crop"
                )

                face_columns = st.columns(
                    len(
                        visualization_items
                    )
                )

                for column, item in zip(
                    face_columns,
                    visualization_items
                ):

                    with column:

                        st.image(
                            item["face"],
                            caption=(
                                f"Face - "
                                f"Frame "
                                f"{item['frame_index']}"
                            ),
                            use_container_width=True
                        )

            else:

                st.warning(
                    "Không có frame visualization."
                )

            # =================================================
            # FRAME SAMPLING INFORMATION
            # =================================================

            st.divider()

            st.subheader(
                "🎞️ Frame được lấy mẫu"
            )

            frame_indices = result[
                "frame_indices"
            ]

            st.code(
                ", ".join(
                    str(int(index))
                    for index in frame_indices
                )
            )

            # =================================================
            # TECHNICAL INFORMATION
            # =================================================

            with st.expander(
                "🔧 Thông tin kỹ thuật"
            ):

                st.write(
                    f"**Model:** "
                    f"Spatial-Temporal Xception + Bi-LSTM"
                )

                st.write(
                    f"**Sequence length:** "
                    f"{SEQUENCE_LENGTH}"
                )

                st.write(
                    f"**Image size:** "
                    f"{IMAGE_SIZE} × {IMAGE_SIZE}"
                )

                st.write(
                    f"**Xception feature:** "
                    f"2048"
                )

                st.write(
                    f"**Bi-LSTM hidden:** "
                    f"{LSTM_HIDDEN_SIZE}"
                )

                st.write(
                    f"**Bi-LSTM layers:** "
                    f"{LSTM_LAYERS}"
                )

                st.write(
                    f"**Normalization:** "
                    f"ImageNet"
                )

                st.write(
                    f"**Threshold:** "
                    f"{THRESHOLD:.2f}"
                )

                st.write(
                    f"**Checkpoint:** "
                    f"{CHECKPOINT_PATH}"
                )

            # =================================================
            # NOTE
            # =================================================

            st.caption(
                "Lưu ý: Confidence là xác suất đầu ra "
                "của mô hình đối với video đang kiểm tra, "
                "không phải độ chính xác tổng thể của mô hình."
            )

        except Exception as error:

            st.error(
                "❌ Đã xảy ra lỗi trong quá trình phân tích."
            )

            st.exception(
                error
            )

        finally:

            # ------------------------------------------------
            # Xóa file tạm
            # ------------------------------------------------

            if (
                temporary_path is not None
                and os.path.exists(
                    temporary_path
                )
            ):

                try:

                    os.remove(
                        temporary_path
                    )

                except OSError:

                    pass

else:

    # ========================================================
    # CHƯA UPLOAD VIDEO
    # ========================================================

    st.info(
        "👆 Hãy chọn một video để bắt đầu phân tích."
    )

    st.write(
        "Hệ thống sẽ lấy mẫu 15 frame, "
        "phát hiện khuôn mặt bằng MTCNN, "
        "trích xuất đặc trưng bằng Xception "
        "và phân tích quan hệ thời gian bằng Bi-LSTM."
    )