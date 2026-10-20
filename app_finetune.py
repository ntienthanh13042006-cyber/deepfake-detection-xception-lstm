# -*- coding: utf-8 -*-
"""
ỨNG DỤNG PHÁT HIỆN VIDEO DEEPFAKE - UNIFIED FORENSIC UI
========================================================

Tích hợp đồng thời 2 chế độ:
1) Upload video: phân tích toàn timeline bằng 1 Global clip + 6 Segment clips.
2) Webcam: tích lũy bằng chứng tối thiểu 10s, tối đa 20s, sau đó ensemble 5 temporal
   windows và khóa kết luận khi bằng chứng đủ ổn định.

Model giữ nguyên:
    Fine-tuned Xception + Bi-LSTM / Celeb-DF v2

Lưu ý nghiên cứu:
- Không retrain model trong file UI.
- Ngưỡng model 0.98 là ngưỡng đã chọn từ Validation của thí nghiệm fine-tuning.
- Lớp quyết định multi-clip / webcam là policy vận hành, cần được đánh giá riêng
  trên Validation trước khi dùng làm cấu hình thực nghiệm cuối cùng.

Chạy bằng Python 3.12 của project:
    .\\.venv312\\Scripts\\python.exe -m streamlit run D:\\Project\\src_finetune\\app_finetune.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

# ============================================================
# 1. PATH / MODEL CONFIG
# ============================================================
PROJECT_DIR = Path(r"D:\Project")
SRC_FINETUNE_DIR = PROJECT_DIR / "src_finetune"
SRC_DIR = PROJECT_DIR / "src"
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints_celebdf_finetune" / "best_model.pth"

RESULTS_DIR = PROJECT_DIR / "results_web_forensic"
UPLOAD_RESULTS_DIR = RESULTS_DIR / "uploads"
WEBCAM_RESULTS_DIR = RESULTS_DIR / "webcam"
HISTORY_PATH = RESULTS_DIR / "history.json"

for directory in (RESULTS_DIR, UPLOAD_RESULTS_DIR, WEBCAM_RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

for candidate in (SRC_FINETUNE_DIR, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

SEQUENCE_LENGTH = 15
IMAGE_SIZE = 224
MARGIN = 20
MODEL_THRESHOLD = 0.98

# Policy quyết định ở mức clip/video.
SUSPICION_THRESHOLD = 0.83
STRONG_FAKE_THRESHOLD = 0.98
REAL_CONFIRM_THRESHOLD = 0.20
FAKE_SUPPORT_MIN = 0.90
REAL_SUPPORT_MAX = 0.35
REQUIRED_AGREEMENT = 4
MAX_MAD = 0.10

# Upload: Global + 6 segment.
NUM_SEGMENTS = 6
GLOBAL_WEIGHT = 2.0

# Webcam.
WEBCAM_MIN_SECONDS = 10.0
WEBCAM_MAX_SECONDS = 20.0
WEBCAM_RECHECK_EVERY = 5.0
WEBCAM_TEMPORAL_RANGES = (
    (0.00, 1.00),
    (0.05, 0.95),
    (0.00, 0.90),
    (0.10, 1.00),
    (0.15, 0.95),
)
PROCESS_EVERY_N_FRAMES = 3
MAX_PROCESS_WIDTH = 960
MAX_FACE_SAMPLES = 300
FACE_GAP_RESET_SECONDS = 2.5
MAX_HISTORY = 100
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# WebRTC callback chạy ở thread riêng; dùng lock riêng cho model/MTCNN.
MODEL_LOCK = threading.RLock()

# ============================================================
# 2. STREAMLIT CONFIG / LIGHT RED FORENSIC UI
# ============================================================
st.set_page_config(
    page_title="Deepfake Forensic Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --red: #b91c1c;
            --red-dark: #7f1d1d;
            --red-deep: #5f1515;
            --red-soft: #fef2f2;
            --red-border: #fecaca;
            --black: #111111;
            --black-2: #1f2937;
            --gray: #374151;
            --gray-soft: #f3f4f6;
            --border: #d1d5db;
            --surface: #ffffff;
            --success: #166534;
            --success-soft: #f0fdf4;
            --success-border: #bbf7d0;
            --warn: #92400e;
            --warn-soft: #fffbeb;
            --warn-border: #fde68a;
            --bg: #f4f5f7;
        }

        .stApp {
            background: var(--bg);
            color: var(--black) !important;
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                BlinkMacSystemFont, "Segoe UI", sans-serif !important;
            color: var(--black) !important;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 1rem;
            padding-bottom: 3rem;
        }

        /* Bảo đảm chữ luôn đủ tương phản */
        p, span, label, div, small, li, td, th, textarea, input {
            color: var(--black);
        }

        .forensic-header {
            background: linear-gradient(135deg, #ffffff 0%, #fff6f6 100%);
            border: 1px solid var(--red-border);
            border-left: 8px solid var(--red);
            border-radius: 20px;
            padding: 25px 30px;
            box-shadow: 0 12px 30px rgba(127, 29, 29, .08);
            margin-bottom: 18px;
        }

        .kicker {
            color: var(--red-dark) !important;
            font-size: .75rem;
            font-weight: 900;
            letter-spacing: .15em;
            text-transform: uppercase;
        }

        .title {
            color: var(--black) !important;
            font-size: 2.2rem;
            font-weight: 900;
            line-height: 1.15;
            margin: 6px 0;
        }

        .subtitle {
            color: var(--black-2) !important;
            font-size: .98rem;
            line-height: 1.6;
            max-width: 1100px;
        }

        .hero-tag {
            display: inline-block;
            margin-top: 12px;
            background: var(--red-soft);
            color: var(--red-dark) !important;
            border: 1px solid var(--red-border);
            border-radius: 999px;
            padding: 6px 12px;
            font-size: .76rem;
            font-weight: 900;
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 17px;
            padding: 18px;
            box-shadow: 0 7px 20px rgba(17,24,39,.05);
            margin: 8px 0 14px 0;
        }

        .card-title {
            color: var(--black) !important;
            margin: 0 0 8px 0;
            font-weight: 900;
            font-size: 1.02rem;
        }

        .muted {
            color: var(--gray) !important;
            font-size: .88rem;
            line-height: 1.55;
        }

        .decision {
            border-radius: 18px;
            padding: 22px;
            border: 1px solid var(--border);
            background: #ffffff;
            box-shadow: 0 10px 26px rgba(17,24,39,.07);
        }

        .decision.fake {
            border-left: 8px solid var(--red);
            background: #fff6f6;
        }

        .decision.real {
            border-left: 8px solid var(--success);
            background: #f7fff9;
        }

        .decision.review {
            border-left: 8px solid #d97706;
            background: #fffdf5;
        }

        .decision-main {
            font-size: 1.75rem;
            font-weight: 950;
            margin: 4px 0;
            color: var(--black) !important;
        }

        .decision.fake .decision-main { color: var(--red-dark) !important; }
        .decision.real .decision-main { color: var(--success) !important; }
        .decision.review .decision-main { color: var(--warn) !important; }

        .note, .danger-note, .success-note, .warning-note {
            border-radius: 13px;
            padding: 13px 15px;
            color: var(--black) !important;
            border: 1px solid var(--border);
            background: #ffffff;
        }
        .danger-note { background: var(--red-soft); border-color: var(--red-border); }
        .success-note { background: var(--success-soft); border-color: var(--success-border); }
        .warning-note { background: var(--warn-soft); border-color: var(--warn-border); }

        .case-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            background: var(--black);
            color: white !important;
            font-size: .73rem;
            font-weight: 800;
        }

        .red-line {
            height: 4px;
            background: var(--red);
            border-radius: 99px;
            margin: 10px 0 16px 0;
        }

        .stTabs [data-baseweb="tab"] {
            color: var(--black) !important;
            font-weight: 900 !important;
        }

        .stTabs [aria-selected="true"] {
            color: var(--red) !important;
        }

        div[role="radiogroup"] {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 15px;
            padding: 6px;
            box-shadow: 0 6px 16px rgba(17,24,39,.05);
        }
        div[role="radiogroup"] label {
            color: var(--black) !important;
            font-weight: 900 !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: #ffffff !important;
            border: 2px dashed #ef4444 !important;
            border-radius: 16px !important;
            padding: 18px !important;
        }
        [data-testid="stFileUploaderDropzone"] * {
            color: var(--black) !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] {
            color: var(--black) !important;
        }

        div.stButton > button,
        div[data-testid="stFileUploader"] button,
        .stDownloadButton button {
            border-radius: 12px !important;
            font-weight: 900 !important;
            min-height: 44px !important;
            border: 1px solid var(--border) !important;
        }

        div.stButton > button[kind="primary"] {
            background: var(--red) !important;
            border-color: var(--red) !important;
            color: #ffffff !important;
        }

        div.stButton > button[kind="primary"] p,
        div.stButton > button[kind="primary"] span {
            color: #ffffff !important;
        }

        div.stButton > button[kind="primary"]:hover {
            background: var(--red-dark) !important;
            border-color: var(--red-dark) !important;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] * { color: var(--black) !important; }
        [data-testid="stMetricValue"] { color: var(--black) !important; }
        [data-testid="stMetricLabel"] { color: var(--gray) !important; }
        [data-testid="stDataFrame"] * { color: var(--black) !important; }

        .small-red {
            color: var(--red-dark) !important;
            font-weight: 900;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 3. LOAD MODEL
# ============================================================
try:
    from inference_pipeline_finetune import FineTunePredictor
except Exception as exc:
    st.error("Không thể import inference_pipeline_finetune.py")
    with st.expander("Chi tiết lỗi kỹ thuật"):
        st.exception(exc)
    st.stop()


@st.cache_resource(show_spinner="Đang khởi tạo Xception + Bi-LSTM và MTCNN...")
def load_predictor() -> FineTunePredictor:
    return FineTunePredictor(
        checkpoint_path=CHECKPOINT_PATH,
        threshold=MODEL_THRESHOLD,
    )


try:
    predictor = load_predictor()
except Exception as exc:
    st.error("Không thể tải checkpoint mô hình.")
    with st.expander("Chi tiết lỗi kỹ thuật"):
        st.exception(exc)
    st.stop()

# ============================================================
# 4. COMMON HELPERS
# ============================================================
def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def sigmoid(x: float) -> float:
    x = float(np.clip(x, -80.0, 80.0))
    return 1.0 / (1.0 + math.exp(-x))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mad(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def read_frame_at(cap: cv2.VideoCapture, index: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    return frame if ok and frame is not None else None


def get_video_meta(video_path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video: {video_path}")
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = safe_float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frames / fps if fps > 0 else 0.0
        if frames < SEQUENCE_LENGTH:
            raise ValueError(
                f"Video có {frames} frame, không đủ {SEQUENCE_LENGTH} frame tối thiểu."
            )
        return {
            "total_frames": frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_seconds": duration,
        }
    finally:
        cap.release()


def face_tensor_to_uint8(face: torch.Tensor) -> np.ndarray:
    """Chuyển tensor MTCNN (3,H,W) về H,W,3 uint8."""
    if not isinstance(face, torch.Tensor):
        raise TypeError(f"MTCNN output không phải Tensor: {type(face)}")
    if face.ndim != 3 or face.shape[0] != 3:
        raise ValueError(f"MTCNN output shape không hợp lệ: {tuple(face.shape)}")
    face_np = face.permute(1, 2, 0).detach().cpu().numpy()
    # post_process=False trong predictor -> dữ liệu 0..255.
    return np.clip(face_np, 0, 255).astype(np.uint8)


def crop_face(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Face crop đúng cấu hình MTCNN của pipeline, không gọi method không tồn tại."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb)
    with torch.no_grad():
        face = predictor.preprocessor.mtcnn(image_pil)
    if face is None:
        return None
    out = face_tensor_to_uint8(face)
    expected = (IMAGE_SIZE, IMAGE_SIZE, 3)
    return out if out.shape == expected else None


def detect_box_and_probability(
    frame_bgr: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Lấy bbox + confidence. Dùng detect() của cùng MTCNN instance."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb)
    try:
        with torch.no_grad():
            boxes, probs = predictor.preprocessor.mtcnn.detect(image_pil)
        if boxes is None or probs is None or len(boxes) == 0:
            return None, None
        return np.asarray(boxes[0], dtype=np.float32), safe_float(probs[0], 0.0)
    except Exception:
        return None, None


def run_sequence(sequence: np.ndarray) -> Dict[str, float]:
    expected = (SEQUENCE_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 3)
    if sequence.shape != expected:
        raise ValueError(f"Sequence shape không hợp lệ: {sequence.shape}; cần {expected}")
    result = predictor.predict_sequence(sequence)
    fake = float(np.clip(safe_float(result.get("fake_probability")), 0.0, 1.0))
    return {"fake_probability": fake, "real_probability": 1.0 - fake}


# ============================================================
# 5. UPLOAD VIDEO - MULTI-CLIP FORENSIC INFERENCE
# ============================================================
def global_indices(total: int) -> np.ndarray:
    return np.linspace(0, total - 1, SEQUENCE_LENGTH, dtype=int)


def segment_indices(total: int, segment_id: int) -> np.ndarray:
    edges = np.linspace(0, total, NUM_SEGMENTS + 1)
    start = int(math.floor(edges[segment_id]))
    end = int(math.ceil(edges[segment_id + 1])) - 1
    start = max(0, min(start, total - 1))
    end = max(start, min(end, total - 1))
    return np.linspace(start, end, SEQUENCE_LENGTH, dtype=int)


def analyze_upload_clip(
    cap: cv2.VideoCapture,
    indices: np.ndarray,
    kind: str,
    clip_id: int,
) -> Dict[str, Any]:
    faces: List[np.ndarray] = []
    detected = 0
    reused = 0
    failures = 0
    last_face: Optional[np.ndarray] = None

    for index in indices:
        frame = read_frame_at(cap, int(index))
        if frame is None:
            failures += 1
            continue
        try:
            face = crop_face(frame)
        except Exception:
            face = None
        if face is not None:
            faces.append(face)
            last_face = face.copy()
            detected += 1
        elif last_face is not None:
            # Chỉ dùng reuse để duy trì sequence của video upload, và ghi lại tỷ lệ reuse.
            faces.append(last_face.copy())
            reused += 1

    if not faces:
        raise RuntimeError(f"Clip {kind} {clip_id}: không phát hiện được khuôn mặt.")

    while len(faces) < SEQUENCE_LENGTH:
        faces.append(faces[-1].copy())
        reused += 1

    with MODEL_LOCK, torch.no_grad():
        pred = run_sequence(np.asarray(faces[:SEQUENCE_LENGTH], dtype=np.uint8))

    pred.update(
        {
            "kind": kind,
            "clip_id": clip_id,
            "frame_indices": [int(x) for x in indices],
            "detected_faces": detected,
            "reused_faces": reused,
            "reuse_ratio": reused / SEQUENCE_LENGTH,
            "read_failures": failures,
        }
    )
    return pred


def decide_upload(clip_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tổng hợp clip bằng logit-space để không bị lệch quá mạnh bởi score 0/1."""
    scores = [float(np.clip(safe_float(x["fake_probability"]), 1e-6, 1 - 1e-6)) for x in clip_results]
    logits = [math.log(s / (1.0 - s)) for s in scores]
    weights = [GLOBAL_WEIGHT if x["kind"] == "GLOBAL" else 1.0 for x in clip_results]
    agg_logit = float(np.average(np.asarray(logits), weights=np.asarray(weights)))
    agg_fake = sigmoid(agg_logit)

    median = float(np.median(scores))
    dispersion = mad(scores)
    fake_votes = sum(s >= FAKE_SUPPORT_MIN for s in scores)
    real_votes = sum(s <= REAL_SUPPORT_MAX for s in scores)
    strong_fake_votes = sum(s >= STRONG_FAKE_THRESHOLD for s in scores)

    if agg_fake >= STRONG_FAKE_THRESHOLD and fake_votes >= REQUIRED_AGREEMENT and strong_fake_votes >= 3:
        decision = "FAKE"
        reason = "Điểm tổng hợp cao, nhiều temporal clip đồng thuận và có bằng chứng Fake mạnh."
    elif agg_fake <= REAL_CONFIRM_THRESHOLD and real_votes >= REQUIRED_AGREEMENT and dispersion <= MAX_MAD:
        decision = "REAL"
        reason = "Điểm Fake tổng hợp thấp, các temporal clip nhất quán và không có tín hiệu Fake mạnh."
    else:
        decision = "REVIEW"
        reason = "Bằng chứng giữa các temporal clip chưa đủ nhất quán; hệ thống không ép thành REAL/FAKE."

    low_quality = sum(
        int(x["detected_faces"]) < 12 or float(x["reuse_ratio"]) > 0.20
        for x in clip_results
    )
    if low_quality >= 3:
        decision = "REVIEW"
        reason = "Nhiều temporal clip có chất lượng khuôn mặt thấp; cần kiểm tra lại điều kiện video."

    return {
        "decision": decision,
        "fake_probability": agg_fake,
        "real_probability": 1.0 - agg_fake,
        "median_clip_fake": median,
        "mad": dispersion,
        "fake_votes": fake_votes,
        "real_votes": real_votes,
        "strong_fake_votes": strong_fake_votes,
        "num_clips": len(scores),
        "reason": reason,
    }


def save_upload_report(
    file_name: str,
    file_hash: str,
    meta: Dict[str, Any],
    decision: Dict[str, Any],
    clips: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Path]:
    case_id = datetime.now().strftime("upload_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    report = {
        "case_id": case_id,
        "timestamp": datetime.now().astimezone().isoformat(),
        "file_name": file_name,
        "sha256": file_hash,
        "video": meta,
        "decision": decision,
        "clips": clips,
        "model": {
            "name": "Fine-tuned Xception + Bi-LSTM",
            "dataset": "Celeb-DF v2",
            "sequence_length": SEQUENCE_LENGTH,
            "input_face": "224x224",
            "mtcnn_margin": MARGIN,
            "checkpoint": str(CHECKPOINT_PATH),
            "device": str(predictor.device),
            "checkpoint_epoch": predictor.checkpoint_epoch,
            "checkpoint_val_auc": predictor.checkpoint_val_auc,
        },
        "decision_policy": {
            "num_clips": len(clips),
            "model_threshold": MODEL_THRESHOLD,
            "global_weight": GLOBAL_WEIGHT,
            "fake_support_min": FAKE_SUPPORT_MIN,
            "real_support_max": REAL_SUPPORT_MAX,
            "required_agreement": REQUIRED_AGREEMENT,
            "max_mad": MAX_MAD,
        },
        "note": "Điểm Fake phục vụ sàng lọc; không phải xác suất pháp lý và không thay thế giám định chuyên môn.",
    }
    path = UPLOAD_RESULTS_DIR / f"{case_id}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, path


# ============================================================
# 6. WEBCAM - ACCUMULATE EVIDENCE / LOCK DECISION
# ============================================================
class WebcamState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "lock", threading.RLock()):
            self.generation = getattr(self, "generation", 0) + 1
            self.case_id = datetime.now().strftime("webcam_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
            self.case_started_at: Optional[float] = None
            self.last_face_seen_at: Optional[float] = None
            self.face_buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=MAX_FACE_SAMPLES)
            self.total_frames = 0
            self.processed_frames = 0
            self.detected_faces = 0
            self.missed_faces = 0
            self.last_box: Optional[np.ndarray] = None
            self.last_face_prob: Optional[float] = None
            self.last_frame: Optional[np.ndarray] = None
            self.representative_face: Optional[np.ndarray] = None
            self.brightness_values: Deque[float] = deque(maxlen=120)
            self.sharpness_values: Deque[float] = deque(maxlen=120)
            self.quality_values: Deque[float] = deque(maxlen=120)
            self.next_check = WEBCAM_MIN_SECONDS
            self.inference_running = False
            self.final_ready = False
            self.final_label = "CHƯA CÓ KẾT LUẬN"
            self.final_fake: Optional[float] = None
            self.final_real: Optional[float] = None
            self.last_scores: List[float] = []
            self.last_fake_votes = 0
            self.last_real_votes = 0
            self.last_mad: Optional[float] = None
            self.last_round_decision = ""
            self.last_analysis_elapsed: Optional[float] = None
            self.history_saved = False
            self.saved_record: Optional[Dict[str, Any]] = None
            self.last_error = ""

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            elapsed = 0.0
            if self.case_started_at is not None:
                elapsed = max(0.0, time.monotonic() - self.case_started_at)
            quality = float(np.mean(self.quality_values)) if self.quality_values else 0.0
            brightness = float(np.mean(self.brightness_values)) if self.brightness_values else 0.0
            sharpness = float(np.mean(self.sharpness_values)) if self.sharpness_values else 0.0
            return {
                "case_id": self.case_id,
                "elapsed": elapsed,
                "buffer": len(self.face_buffer),
                "total_frames": self.total_frames,
                "processed": self.processed_frames,
                "detected": self.detected_faces,
                "missed": self.missed_faces,
                "final_ready": self.final_ready,
                "final_label": self.final_label,
                "final_fake": self.final_fake,
                "final_real": self.final_real,
                "next_check": self.next_check,
                "inference_running": self.inference_running,
                "last_scores": list(self.last_scores),
                "fake_votes": self.last_fake_votes,
                "real_votes": self.last_real_votes,
                "mad": self.last_mad,
                "round_decision": self.last_round_decision,
                "last_analysis_elapsed": self.last_analysis_elapsed,
                "quality": quality,
                "brightness": brightness,
                "sharpness": sharpness,
                "history_saved": self.history_saved,
                "saved_record": dict(self.saved_record) if self.saved_record else None,
                "last_error": self.last_error,
                "last_box": self.last_box,
                "last_face_prob": self.last_face_prob,
            }


def quality_from_frame(
    frame_bgr: np.ndarray,
    box: Optional[np.ndarray],
    detector_prob: Optional[float],
) -> Tuple[float, float, float]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    score = 100.0
    if brightness < 45 or brightness > 215:
        score -= 30.0
    elif brightness < 65 or brightness > 195:
        score -= 12.0

    if sharpness < 30:
        score -= 30.0
    elif sharpness < 70:
        score -= 12.0

    if detector_prob is not None and detector_prob < 0.90:
        score -= 15.0

    if box is not None:
        x1, y1, x2, y2 = [float(v) for v in box]
        h, w = frame_bgr.shape[:2]
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, h * w)
        if area_ratio < 0.04:
            score -= 20.0
        elif area_ratio < 0.08:
            score -= 8.0
    else:
        score -= 35.0

    return float(np.clip(score, 0, 100)), brightness, sharpness


def draw_webcam_overlay(frame_bgr: np.ndarray, snap: Dict[str, Any]) -> np.ndarray:
    out = frame_bgr.copy()
    box = snap["last_box"]
    prob = snap["last_face_prob"]

    if box is not None:
        try:
            x1, y1, x2, y2 = [int(v) for v in box]
            h, w = out.shape[:2]
            x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
            y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = "FACE"
            if prob is not None:
                label += f" {prob * 100:.1f}%"
            cv2.putText(
                out,
                label,
                (x1, max(28, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        except Exception:
            pass

    if snap["final_ready"]:
        label = snap["final_label"]
        if label == "FAKE":
            line1 = "KET LUAN DA KHOA: FAKE"
        elif label == "REAL":
            line1 = "KET LUAN DA KHOA: REAL"
        else:
            line1 = "KET LUAN DA KHOA: KHONG DU BANG CHUNG"
        fake = snap["final_fake"]
        real = snap["final_real"]
        line2 = (
            f"Fake {fake * 100:.2f}% | Real {real * 100:.2f}%"
            if fake is not None and real is not None
            else ""
        )
    else:
        elapsed = min(snap["elapsed"], WEBCAM_MAX_SECONDS)
        if elapsed < WEBCAM_MIN_SECONDS:
            line1 = "DANG THU THAP BANG CHUNG"
        else:
            line1 = "DANG XAC MINH THEM"
        line2 = (
            f"Quan sat {elapsed:.1f}/{WEBCAM_MAX_SECONDS:.0f}s | "
            f"Face {snap['buffer']}"
        )

    box_w = min(out.shape[1] - 10, 1160)
    cv2.rectangle(out, (10, 10), (box_w, 110), (15, 23, 42), -1)
    cv2.putText(out, line1, (24, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, line2, (24, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def build_webcam_sequence(
    samples: List[Tuple[float, np.ndarray]],
    a: float,
    b: float,
) -> np.ndarray:
    ts = np.asarray([x[0] for x in samples], dtype=np.float64)
    crops = [x[1] for x in samples]
    start = ts[0] + (ts[-1] - ts[0]) * a
    end = ts[0] + (ts[-1] - ts[0]) * b
    targets = np.linspace(start, end, SEQUENCE_LENGTH)
    idx = [int(np.argmin(np.abs(ts - t))) for t in targets]
    return np.asarray([crops[i] for i in idx], dtype=np.uint8)


def analyze_webcam_evidence(samples: List[Tuple[float, np.ndarray]]) -> Dict[str, Any]:
    scores: List[float] = []
    with MODEL_LOCK, torch.no_grad():
        for a, b in WEBCAM_TEMPORAL_RANGES:
            seq = build_webcam_sequence(samples, a, b)
            scores.append(float(run_sequence(seq)["fake_probability"]))

    arr = np.asarray(scores, dtype=np.float64)
    med = float(np.median(arr))
    dispersion = mad(scores)
    fake_votes = int(np.sum(arr >= FAKE_SUPPORT_MIN))
    real_votes = int(np.sum(arr <= REAL_SUPPORT_MAX))

    if fake_votes >= REQUIRED_AGREEMENT and med >= STRONG_FAKE_THRESHOLD and dispersion <= MAX_MAD:
        decision = "FAKE"
    elif real_votes >= REQUIRED_AGREEMENT and med <= REAL_CONFIRM_THRESHOLD and dispersion <= MAX_MAD:
        decision = "REAL"
    else:
        decision = "REVIEW"

    return {
        "decision": decision,
        "fake": med,
        "real": 1.0 - med,
        "scores": scores,
        "fake_votes": fake_votes,
        "real_votes": real_votes,
        "mad": dispersion,
    }


def save_webcam_case(
    state_obj: WebcamState,
    analysis: Dict[str, Any],
    elapsed: float,
) -> Dict[str, Any]:
    case_dir = WEBCAM_RESULTS_DIR / state_obj.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    frame_path: Optional[Path] = None
    face_path: Optional[Path] = None

    with state_obj.lock:
        latest_frame = state_obj.last_frame.copy() if state_obj.last_frame is not None else None
        representative = state_obj.representative_face.copy() if state_obj.representative_face is not None else None
        quality_mean = float(np.mean(state_obj.quality_values)) if state_obj.quality_values else 0.0
        brightness_mean = float(np.mean(state_obj.brightness_values)) if state_obj.brightness_values else 0.0
        sharpness_mean = float(np.mean(state_obj.sharpness_values)) if state_obj.sharpness_values else 0.0
        face_samples = len(state_obj.face_buffer)

    if latest_frame is not None:
        annotated = draw_webcam_overlay(latest_frame, state_obj.snapshot())
        frame_path = case_dir / "evidence_frame.jpg"
        cv2.imwrite(str(frame_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

    if representative is not None:
        face_path = case_dir / "representative_face.jpg"
        cv2.imwrite(
            str(face_path),
            cv2.cvtColor(representative, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )

    record = {
        "case_id": state_obj.case_id,
        "timestamp": datetime.now().astimezone().isoformat(),
        "decision": analysis["decision"],
        "fake_probability": round(float(analysis["fake"]), 8),
        "real_probability": round(float(analysis["real"]), 8),
        "observation_seconds": round(float(elapsed), 3),
        "face_samples": face_samples,
        "temporal_samples": len(analysis["scores"]),
        "fake_agreement": int(analysis["fake_votes"]),
        "real_agreement": int(analysis["real_votes"]),
        "median_abs_deviation": round(float(analysis["mad"]), 8),
        "sample_scores": [round(float(x), 8) for x in analysis["scores"]],
        "quality_mean": round(quality_mean, 3),
        "brightness_mean": round(brightness_mean, 3),
        "sharpness_mean": round(sharpness_mean, 3),
        "evidence_frame": str(frame_path) if frame_path else None,
        "face_crop": str(face_path) if face_path else None,
        "model": "Fine-tuned Xception + Bi-LSTM / Celeb-DF v2",
        "note": "Case-level evidence stabilization; không phải liveness/anti-spoofing và không phải chứng nhận pháp lý.",
    }

    existing: List[Dict[str, Any]] = []
    if HISTORY_PATH.exists():
        try:
            loaded = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = loaded
        except Exception:
            existing = []

    existing.append(record)
    HISTORY_PATH.write_text(
        json.dumps(existing[-MAX_HISTORY:], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (case_dir / "case_report.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return record


# ============================================================
# 7. HEADER / SIDEBAR
# ============================================================
st.markdown(
    """
    <div class="forensic-header">
        <div class="kicker">INFORMATION SECURITY · DIGITAL FORENSICS</div>
        <div class="title">HỆ THỐNG PHÁT HIỆN VIDEO DEEPFAKE</div>
        <div class="subtitle">
            Một giao diện duy nhất cho <b>Upload Video</b> và <b>Webcam</b>.
            Hệ thống phân tích nhiều temporal clip, đánh giá chất lượng đầu vào,
            tránh kết luận quá sớm, khóa kết luận khi đủ bằng chứng và lưu lại Case
            để người dùng có thể kiểm tra lại.
        </div>
        <div class="hero-tag">FINE-TUNED XCEPTION + BI-LSTM · CELEB-DF V2 · FORENSIC SCREENING</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ⚙️ CẤU HÌNH HỆ THỐNG")
    st.markdown(
        f"""
        <div class="card">
            <div class="muted">MODEL</div>
            <div class="card-title">Fine-tuned Xception + Bi-LSTM</div>
            <div class="muted">Celeb-DF v2</div>
        </div>
        <div class="card">
            <div class="muted">FACE INPUT</div>
            <div class="card-title">{IMAGE_SIZE} × {IMAGE_SIZE}px</div>
            <div class="muted">Sequence = {SEQUENCE_LENGTH} face frames</div>
        </div>
        <div class="card">
            <div class="muted">MODEL THRESHOLD</div>
            <div class="card-title">{MODEL_THRESHOLD:.2f}</div>
            <div class="muted">Ngưỡng đã chọn từ Validation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔄 TẢI LẠI MÔ HÌNH", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.markdown("### 📌 Cách sử dụng webcam")
    st.markdown(
        """
        **1.** Cho phép trình duyệt sử dụng camera.  
        **2.** Để mặt đủ lớn, gần chính diện.  
        **3.** Tránh ngược sáng, rung và che khuất.  
        **4.** Đợi hệ thống quan sát tối thiểu 10 giây.  
        **5.** Không xem `REVIEW` là `REAL`.  
        **6.** Khi đưa video Deepfake qua webcam, nên tránh moiré / phản chiếu màn hình.
        """
    )

    st.divider()
    st.markdown("### 🔬 Nguyên tắc kết luận")
    st.markdown(
        """
        Hệ thống ưu tiên **bằng chứng ở mức video/case**, không dùng một frame đơn lẻ để kết luận.
        Khi bằng chứng không đủ nhất quán, hệ thống giữ trạng thái **CẦN KIỂM TRA THÊM**.
        """
    )

    st.divider()
    st.caption(
        "Webcam là mô-đun suy luận thời gian thực / trình diễn. Không phải hệ thống liveness hoặc anti-spoofing chuyên dụng."
    )

# ============================================================
# 8. SYSTEM STATUS
# ============================================================
ststatus = st.columns(4)
with ststatus[0]:
    st.metric("Thiết bị", str(predictor.device).upper())
with ststatus[1]:
    epoch_text = str(predictor.checkpoint_epoch) if predictor.checkpoint_epoch is not None else "N/A"
    st.metric("Checkpoint epoch", epoch_text)
with ststatus[2]:
    auc = predictor.checkpoint_val_auc
    st.metric("Validation ROC-AUC", f"{auc:.4f}" if auc is not None else "N/A")
with ststatus[3]:
    st.metric("Face / sequence", f"{SEQUENCE_LENGTH} × {IMAGE_SIZE}px")

st.markdown('<div class="red-line"></div>', unsafe_allow_html=True)

# ============================================================
# 9. MODE SELECTOR - CHỈ CHẠY MỘT CHẾ ĐỘ MỖI LẦN
# ============================================================
# Quan trọng: st.tabs() vẫn thực thi nội dung của cả hai tab.
# Điều này có thể làm lỗi webcam (av/streamlit-webrtc) ảnh hưởng UX Upload.
# Dùng radio ngang giúp Upload và Webcam hoàn toàn độc lập.
mode = st.radio(
    "Chế độ phân tích",
    ["📁 UPLOAD VIDEO", "🎥 WEBCAM TRỰC TIẾP"],
    horizontal=True,
    label_visibility="collapsed",
    key="analysis_mode_v10",
)

# ============================================================
# 9A. UPLOAD VIDEO
# ============================================================
if mode == "📁 UPLOAD VIDEO":
    st.markdown(
        """
        <div class="card">
            <div class="card-title">📁 Phân tích video đã có sẵn</div>
            <div class="muted">
                Chọn video từ máy tính, kiểm tra thông tin video rồi nhấn
                <b>PHÂN TÍCH TOÀN VIDEO</b>. Hệ thống quét <b>1 Global clip + 6 Segment clips</b>
                để giảm phụ thuộc vào một đoạn ngắn duy nhất.
            </div>
            <div class="hero-tag">Hỗ trợ tối đa 1024 MB / video</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Chọn video cần phân tích",
        type=sorted(x.lstrip(".") for x in SUPPORTED_EXTENSIONS),
        accept_multiple_files=False,
        max_upload_size=1024,
        key="deepfake_video_uploader_v10",
        help=(
            "Hỗ trợ MP4/AVI/MOV/MKV/WEBM. Giới hạn giao diện: 1024 MB. "
            "Nên dùng video có khuôn mặt rõ, đủ sáng và ít rung."
        ),
    )

    st.session_state.setdefault("upload_result_v10", None)
    st.session_state.setdefault("upload_hash_v10", None)

    if uploaded is None:
        st.markdown(
            "<div class='note'><b>Chưa chọn video.</b> Bấm vào vùng tải tệp ở trên và chọn một video từ máy tính.</div>",
            unsafe_allow_html=True,
        )

    if uploaded is not None:
        buffer = uploaded.getbuffer()
        file_size_mb = len(buffer) / (1024 * 1024)
        file_hash = hashlib.sha256(buffer).hexdigest()

        # Video mới -> xóa ngay kết quả video trước.
        if st.session_state.upload_hash_v10 != file_hash:
            st.session_state.upload_hash_v10 = file_hash
            st.session_state.upload_result_v10 = None

        suffix = Path(uploaded.name).suffix.lower()
        temp_path: Optional[Path] = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(buffer)
                temp_path = Path(tmp.name)
            meta = get_video_meta(temp_path)
        except Exception as exc:
            meta = None
            st.error(f"Không đọc được video: {exc}")

        if meta is not None and temp_path is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Thời lượng", f"{meta['duration_seconds']:.2f}s")
            c2.metric("FPS", f"{meta['fps']:.2f}")
            c3.metric("Resolution", f"{meta['width']}×{meta['height']}")
            c4.metric("Frame", str(meta["total_frames"]))

            # Video nhỏ: preview trực tiếp. Video lớn: không nhân bản dữ liệu lên UI.
            if file_size_mb <= 250:
                st.video(bytes(buffer))
            else:
                st.info(
                    f"Video {file_size_mb:.1f} MB. Đã nhận tệp thành công; để tránh tải lại payload lớn trong trình duyệt, phần preview được tắt. "
                    "Bạn vẫn có thể phân tích bình thường."
                )

            st.markdown(
                f"<div class='note'><b>Tệp đã nhận:</b> {uploaded.name} · {file_size_mb:.1f} MB · <b>SHA-256:</b> {file_hash[:20]}…</div>",
                unsafe_allow_html=True,
            )

            b1, b2 = st.columns([4, 1])
            with b1:
                run_upload = st.button(
                    "🔎 PHÂN TÍCH TOÀN VIDEO",
                    type="primary",
                    use_container_width=True,
                )
            with b2:
                clear_upload = st.button("Xóa kết quả", use_container_width=True)

            if clear_upload:
                st.session_state.upload_result_v10 = None
                st.rerun()

            if run_upload:
                try:
                    with st.status("Đang phân tích toàn bộ timeline...", expanded=True) as status:
                        clips: List[Dict[str, Any]] = []
                        st.write("1/3 · Lấy mẫu Global clip")
                        cap = cv2.VideoCapture(str(temp_path))
                        if not cap.isOpened():
                            raise RuntimeError("Không mở được video.")
                        try:
                            clips.append(
                                analyze_upload_clip(
                                    cap,
                                    global_indices(meta["total_frames"]),
                                    "GLOBAL",
                                    0,
                                )
                            )
                            st.write("2/3 · Lấy mẫu 6 Segment clips + MTCNN")
                            for seg in range(NUM_SEGMENTS):
                                clips.append(
                                    analyze_upload_clip(
                                        cap,
                                        segment_indices(meta["total_frames"], seg),
                                        "SEGMENT",
                                        seg + 1,
                                    )
                                )
                        finally:
                            cap.release()

                        st.write("3/3 · Xception + Bi-LSTM + ensemble")
                        decision = decide_upload(clips)
                        report, report_path = save_upload_report(
                            uploaded.name,
                            file_hash,
                            meta,
                            decision,
                            clips,
                        )
                        report["report_path"] = str(report_path)
                        st.session_state.upload_result_v10 = report
                        status.update(
                            label="Phân tích hoàn tất",
                            state="complete",
                            expanded=False,
                        )
                except Exception as exc:
                    st.error("Phân tích video thất bại.")
                    with st.expander("Chi tiết lỗi kỹ thuật"):
                        st.exception(exc)

            result = st.session_state.upload_result_v10
            if result is not None:
                decision = result["decision"]
                d = decision["decision"]
                cls = "fake" if d == "FAKE" else "real" if d == "REAL" else "review"
                label = (
                    "🚨 VIDEO FAKE"
                    if d == "FAKE"
                    else "✅ VIDEO REAL"
                    if d == "REAL"
                    else "⚠️ CẦN KIỂM TRA THÊM"
                )

                st.markdown(
                    f"""
                    <div class="decision {cls}">
                        <div class="muted">KẾT LUẬN Ở MỨC VIDEO</div>
                        <div class="decision-main">{label}</div>
                        <div class="muted">{decision['reason']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Điểm Fake ensemble", f"{decision['fake_probability'] * 100:.2f}%")
                m2.metric("Điểm Real ensemble", f"{decision['real_probability'] * 100:.2f}%")
                m3.metric("Fake agreement", f"{decision['fake_votes']}/{decision['num_clips']}")
                m4.metric("MAD", f"{decision['mad']:.4f}")

                if d == "FAKE":
                    st.markdown(
                        '<div class="danger-note"><b>⚠️ Cảnh báo:</b> video có tín hiệu giả mạo đáng chú ý theo mô hình. Hãy đối chiếu thêm bằng chứng độc lập nếu dùng cho mục đích điều tra.</div>',
                        unsafe_allow_html=True,
                    )
                elif d == "REAL":
                    st.markdown(
                        '<div class="success-note"><b>ℹ️ Lưu ý:</b> kết quả này chỉ phản ánh bằng chứng mà mô hình quan sát được; không phải xác nhận tuyệt đối nội dung là thật.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="warning-note"><b>⚠️ Không đủ bằng chứng:</b> temporal clips còn mâu thuẫn hoặc chất lượng face chưa đủ tốt. Không nên coi REVIEW là REAL.</div>',
                        unsafe_allow_html=True,
                    )

                st.subheader("Phân tích theo temporal clip")
                rows: List[Dict[str, str]] = []
                for clip in result["clips"]:
                    p = safe_float(clip["fake_probability"])
                    local = (
                        "FAKE mạnh"
                        if p >= STRONG_FAKE_THRESHOLD
                        else "Nghi ngờ FAKE"
                        if p >= SUSPICION_THRESHOLD
                        else "Ủng hộ REAL"
                        if p <= REAL_CONFIRM_THRESHOLD
                        else "Mâu thuẫn"
                    )
                    rows.append(
                        {
                            "Clip": f"{clip['kind']} {clip['clip_id']}",
                            "Fake score": f"{p * 100:.3f}%",
                            "Trạng thái": local,
                            "Face": f"{clip['detected_faces']}/{SEQUENCE_LENGTH}",
                            "Reuse": f"{clip['reuse_ratio'] * 100:.1f}%",
                        }
                    )
                st.dataframe(rows, use_container_width=True, hide_index=True)

                st.download_button(
                    "⬇️ TẢI BÁO CÁO CASE JSON",
                    data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name=f"{result['case_id']}.json",
                    mime="application/json",
                    use_container_width=True,
                )

                st.caption(
                    f"Case: {result['case_id']} · SHA-256: {result['sha256'][:32]}… · Report: {result['report_path']}"
                )

# ============================================================
# 9B. WEBCAM
# ============================================================
if mode == "🎥 WEBCAM TRỰC TIẾP":
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Webcam — tích lũy bằng chứng trước khi kết luận</div>
            <div class="muted">
                Hệ thống không kết luận từ một frame đơn lẻ. Webcam quan sát tối thiểu
                <b>{WEBCAM_MIN_SECONDS:.0f} giây</b>, có thể kéo dài đến <b>{WEBCAM_MAX_SECONDS:.0f} giây</b>,
                sau đó chạy {len(WEBCAM_TEMPORAL_RANGES)} temporal windows. Khi đủ bằng chứng,
                kết luận được <b>khóa</b> và lưu thành Case.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Import webcam dependencies riêng để Upload vẫn hoạt động nếu máy chưa cài.
    try:
        import av
        from streamlit_webrtc import webrtc_streamer
    except Exception as exc:
        st.error("Webcam chưa sẵn sàng. Upload video vẫn có thể sử dụng bình thường.")
        st.code(
            r".\.venv312\Scripts\python.exe -m pip install -U av streamlit-webrtc",
            language="powershell",
        )
        with st.expander("Chi tiết lỗi kỹ thuật"):
            st.exception(exc)
    else:
        st.session_state.setdefault("webcam_state_v8", WebcamState())
        webcam_state: WebcamState = st.session_state.webcam_state_v8

        def callback(frame: av.VideoFrame) -> av.VideoFrame:
            """Callback WebRTC: chỉ cập nhật state dùng lock, không gọi st.*."""
            try:
                frame_bgr = frame.to_ndarray(format="bgr24")

                with webcam_state.lock:
                    webcam_state.total_frames += 1
                    frame_no = webcam_state.total_frames

                    # Khi case đã khóa, chỉ vẽ kết luận cũ; không inference thêm.
                    if webcam_state.final_ready:
                        snap = webcam_state.snapshot()
                        return av.VideoFrame.from_ndarray(
                            draw_webcam_overlay(frame_bgr, snap),
                            format="bgr24",
                        )

                if frame_no % PROCESS_EVERY_N_FRAMES != 0:
                    return av.VideoFrame.from_ndarray(
                        draw_webcam_overlay(frame_bgr, webcam_state.snapshot()),
                        format="bgr24",
                    )

                h, w = frame_bgr.shape[:2]
                processed = frame_bgr
                if w > MAX_PROCESS_WIDTH:
                    scale = MAX_PROCESS_WIDTH / float(w)
                    processed = cv2.resize(
                        frame_bgr,
                        (MAX_PROCESS_WIDTH, max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA,
                    )

                # MTCNN crop + bbox. Nếu crop lỗi thì vẫn trả frame về cho người dùng.
                with MODEL_LOCK:
                    face = crop_face(processed)
                    box, face_prob = detect_box_and_probability(processed)

                now = time.monotonic()
                should_check = False
                samples: List[Tuple[float, np.ndarray]] = []
                generation = -1
                analysis_elapsed = 0.0

                with webcam_state.lock:
                    webcam_state.processed_frames += 1
                    webcam_state.last_box = box
                    webcam_state.last_face_prob = face_prob
                    webcam_state.last_frame = processed.copy()

                    quality, brightness, sharpness = quality_from_frame(
                        processed,
                        box,
                        face_prob,
                    )
                    webcam_state.quality_values.append(quality)
                    webcam_state.brightness_values.append(brightness)
                    webcam_state.sharpness_values.append(sharpness)

                    if face is not None:
                        if webcam_state.case_started_at is None:
                            webcam_state.case_started_at = now
                        webcam_state.last_face_seen_at = now
                        webcam_state.detected_faces += 1
                        # Chỉ lưu face thực sự phát hiện được, không nhân bản vào buffer.
                        webcam_state.face_buffer.append((now, face.copy()))
                        webcam_state.representative_face = face.copy()
                    else:
                        webcam_state.missed_faces += 1

                    elapsed = (
                        now - webcam_state.case_started_at
                        if webcam_state.case_started_at is not None
                        else 0.0
                    )

                    gap = (
                        now - webcam_state.last_face_seen_at
                        if webcam_state.last_face_seen_at is not None
                        else 0.0
                    )

                    # Khuôn mặt biến mất lâu -> case mới, không trộn evidence cũ.
                    if (
                        gap >= FACE_GAP_RESET_SECONDS
                        and webcam_state.face_buffer
                        and not webcam_state.final_ready
                        and not webcam_state.inference_running
                    ):
                        webcam_state.reset()
                        webcam_state.last_error = "Khuôn mặt biến mất quá lâu; hệ thống đã tạo Case mới."
                        elapsed = 0.0

                    should_check = (
                        webcam_state.case_started_at is not None
                        and elapsed >= webcam_state.next_check
                        and len(webcam_state.face_buffer) >= SEQUENCE_LENGTH
                        and not webcam_state.inference_running
                        and not webcam_state.final_ready
                    )

                    if should_check:
                        samples = [
                            (float(ts), img.copy())
                            for ts, img in webcam_state.face_buffer
                        ]
                        generation = webcam_state.generation
                        analysis_elapsed = float(elapsed)
                        webcam_state.inference_running = True

                if should_check:
                    try:
                        analysis = analyze_webcam_evidence(samples)

                        with webcam_state.lock:
                            if webcam_state.generation != generation:
                                return av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")

                            webcam_state.last_scores = list(analysis["scores"])
                            webcam_state.last_fake_votes = int(analysis["fake_votes"])
                            webcam_state.last_real_votes = int(analysis["real_votes"])
                            webcam_state.last_mad = float(analysis["mad"])
                            webcam_state.last_round_decision = analysis["decision"]
                            webcam_state.last_analysis_elapsed = analysis_elapsed
                            webcam_state.inference_running = False

                            quality_mean = (
                                float(np.mean(webcam_state.quality_values))
                                if webcam_state.quality_values
                                else 0.0
                            )
                            quality_ok = quality_mean >= 55.0
                            stable = (
                                analysis["decision"] in {"REAL", "FAKE"}
                                and quality_ok
                            )

                            if stable:
                                webcam_state.final_ready = True
                                webcam_state.final_label = analysis["decision"]
                                webcam_state.final_fake = float(analysis["fake"])
                                webcam_state.final_real = float(analysis["real"])

                                if not webcam_state.history_saved:
                                    webcam_state.saved_record = save_webcam_case(
                                        webcam_state,
                                        analysis,
                                        analysis_elapsed,
                                    )
                                    webcam_state.history_saved = True

                            elif analysis_elapsed >= WEBCAM_MAX_SECONDS:
                                review_analysis = dict(analysis)
                                review_analysis["decision"] = "REVIEW"
                                webcam_state.final_ready = True
                                webcam_state.final_label = "REVIEW"
                                webcam_state.final_fake = float(analysis["fake"])
                                webcam_state.final_real = float(analysis["real"])

                                if not webcam_state.history_saved:
                                    webcam_state.saved_record = save_webcam_case(
                                        webcam_state,
                                        review_analysis,
                                        analysis_elapsed,
                                    )
                                    webcam_state.history_saved = True
                            else:
                                webcam_state.next_check = min(
                                    webcam_state.next_check + WEBCAM_RECHECK_EVERY,
                                    WEBCAM_MAX_SECONDS,
                                )

                    except Exception as exc:
                        with webcam_state.lock:
                            webcam_state.inference_running = False
                            webcam_state.last_error = f"Inference lỗi: {exc}"
                            webcam_state.next_check = min(
                                webcam_state.next_check + WEBCAM_RECHECK_EVERY,
                                WEBCAM_MAX_SECONDS,
                            )

                return av.VideoFrame.from_ndarray(
                    draw_webcam_overlay(processed, webcam_state.snapshot()),
                    format="bgr24",
                )

            except Exception as exc:
                with webcam_state.lock:
                    webcam_state.last_error = f"Webcam callback lỗi: {exc}"
                return frame

        def on_video_ended() -> None:
            # Callback lifecycle của streamlit-webrtc; chỉ thao tác state dùng lock.
            with webcam_state.lock:
                webcam_state.reset()

        try:
            ctx = webrtc_streamer(
                key="deepfake-forensic-webcam-v8",
                video_frame_callback=callback,
                on_video_ended=on_video_ended,
                media_stream_constraints={"video": True, "audio": False},
                media_toggle_controls=False,
            )
        except Exception as exc:
            st.error("Không thể khởi tạo webcam WebRTC.")
            with st.expander("Chi tiết lỗi kỹ thuật"):
                st.exception(exc)
        else:
            if not ctx.state.playing:
                if st.button(
                    "🔄 XÓA CASE LIVE / BẮT ĐẦU CASE MỚI",
                    use_container_width=True,
                ):
                    with webcam_state.lock:
                        webcam_state.reset()
                    st.rerun()

            st.markdown("### 📊 Trạng thái phân tích")
            result_box = st.empty()
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            p_class, p_score, p_time, p_quality, p_agree = [
                c.empty() for c in (mc1, mc2, mc3, mc4, mc5)
            ]

            def render_webcam_panel(snap: Dict[str, Any]) -> None:
                if snap["final_ready"]:
                    d = snap["final_label"]
                    if d == "FAKE":
                        result_box.error(
                            f"🚨 KẾT LUẬN ĐÃ KHÓA: VIDEO FAKE\n\n"
                            f"Fake {snap['final_fake'] * 100:.2f}% · Real {snap['final_real'] * 100:.2f}%"
                        )
                    elif d == "REAL":
                        result_box.success(
                            f"✅ KẾT LUẬN ĐÃ KHÓA: VIDEO REAL\n\n"
                            f"Real {snap['final_real'] * 100:.2f}% · Fake {snap['final_fake'] * 100:.2f}%"
                        )
                    else:
                        result_box.warning(
                            f"⚠️ KẾT LUẬN ĐÃ KHÓA: KHÔNG ĐỦ BẰNG CHỨNG\n\n"
                            f"Median P(Fake) {snap['final_fake'] * 100:.2f}%"
                        )

                    p_class.metric(
                        "Kết luận",
                        "FAKE" if d == "FAKE" else "REAL" if d == "REAL" else "REVIEW",
                    )
                    p_score.metric("Điểm Fake", f"{snap['final_fake'] * 100:.2f}%")
                    p_time.metric("Quan sát", f"{snap['elapsed']:.1f}s")
                    p_quality.metric("Chất lượng", f"{snap['quality']:.0f}/100")
                    p_agree.metric("Đồng thuận", f"{max(snap['fake_votes'], snap['real_votes'])}/5")

                    if snap["history_saved"]:
                        st.markdown(
                            f"<div class='success-note'><b>💾 Kết luận đã lưu.</b> "
                            f"Case <span class='case-badge'>{snap['case_id']}</span> · "
                            f"ảnh bằng chứng và báo cáo JSON đã được lưu trên máy.</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    result_box.info(
                        f"⏳ CHƯA CÓ KẾT LUẬN\n\n"
                        f"Quan sát {min(snap['elapsed'], WEBCAM_MAX_SECONDS):.1f}/{WEBCAM_MAX_SECONDS:.0f}s · "
                        f"Face samples {snap['buffer']} · "
                        f"Mốc kế tiếp {min(snap['next_check'], WEBCAM_MAX_SECONDS):.0f}s"
                    )
                    p_class.metric("Trạng thái", "ĐANG XÁC MINH")
                    p_score.metric("Điểm Fake", "—")
                    p_time.metric("Quan sát", f"{snap['elapsed']:.1f}s")
                    p_quality.metric("Chất lượng", f"{snap['quality']:.0f}/100")
                    p_agree.metric("Face samples", str(snap["buffer"]))

                    if snap["elapsed"] < WEBCAM_MIN_SECONDS:
                        st.markdown(
                            f"<div class='note'><b>Thu thập bằng chứng:</b> hệ thống chưa được phép kết luận trước {WEBCAM_MIN_SECONDS:.0f} giây.</div>",
                            unsafe_allow_html=True,
                        )
                    elif snap["inference_running"]:
                        st.markdown(
                            "<div class='note'><b>Đang phân tích:</b> 5 temporal windows đang được đưa qua Xception + Bi-LSTM.</div>",
                            unsafe_allow_html=True,
                        )
                    elif snap["last_round_decision"] == "REVIEW":
                        st.markdown(
                            "<div class='warning-note'><b>Bằng chứng chưa ổn định.</b> Hệ thống sẽ tiếp tục quan sát thay vì tự động chuyển sang REAL.</div>",
                            unsafe_allow_html=True,
                        )

                if snap["last_error"]:
                    st.warning(snap["last_error"])

            render_webcam_panel(webcam_state.snapshot())

            # Không dùng while-loop vì nó khóa toàn bộ script Streamlit.
            # Fragment chỉ rerun phần trạng thái webcam trong khi WebRTC đang phát.
            try:
                @st.fragment(run_every=0.5)
                def webcam_status_fragment() -> None:
                    render_webcam_panel(webcam_state.snapshot())
            except Exception:
                webcam_status_fragment = None

            if ctx.state.playing and webcam_status_fragment is not None:
                webcam_status_fragment()

            st.markdown("### 🔬 Xử lý các thách thức môi trường")
            q1, q2, q3 = st.columns(3)
            with q1:
                st.markdown(
                    '<div class="card"><div class="card-title">💡 Ánh sáng</div><div class="muted">Theo dõi brightness để cảnh báo khi mặt quá tối hoặc quá sáng.</div></div>',
                    unsafe_allow_html=True,
                )
            with q2:
                st.markdown(
                    '<div class="card"><div class="card-title">📷 Độ nét / rung</div><div class="muted">Theo dõi sharpness và không khóa kết luận khi chất lượng đầu vào không đạt mức tối thiểu.</div></div>',
                    unsafe_allow_html=True,
                )
            with q3:
                st.markdown(
                    '<div class="card"><div class="card-title">🧑 Khuôn mặt</div><div class="muted">MTCNN dùng cấu hình 224×224, margin 20 và sequence 15 face frames.</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("### 🗂️ Lịch sử Case webcam")
            if HISTORY_PATH.exists():
                try:
                    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
                    if not isinstance(history, list):
                        history = []
                except Exception:
                    history = []
            else:
                history = []

            if history:
                rows: List[Dict[str, str]] = []
                for item in reversed(history[-10:]):
                    decision_text = item.get("decision", "")
                    agreement = (
                        item.get("fake_agreement", 0)
                        if decision_text == "FAKE"
                        else item.get("real_agreement", 0)
                    )
                    rows.append(
                        {
                            "Thời gian": str(item.get("timestamp", ""))[:19].replace("T", " "),
                            "Case": item.get("case_id", ""),
                            "Kết luận": "CẦN KIỂM TRA" if decision_text == "REVIEW" else decision_text,
                            "Fake": f"{safe_float(item.get('fake_probability')) * 100:.2f}%",
                            "Quan sát": f"{safe_float(item.get('observation_seconds')):.1f}s",
                            "Agreement": f"{agreement}/5",
                        }
                    )
                st.dataframe(rows, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ TẢI LỊCH SỬ CASE JSON",
                    data=json.dumps(history, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name="deepfake_webcam_history.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.info("Chưa có Case webcam nào được lưu.")

# ============================================================
# 10. FOOTER
# ============================================================
st.markdown('<div class="red-line"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="muted">Deepfake Forensic Detection · Fine-tuned Xception + Bi-LSTM · Celeb-DF v2 · Upload + Webcam · Kết luận hỗ trợ sàng lọc, không thay thế giám định chuyên môn.</div>',
    unsafe_allow_html=True,
)
