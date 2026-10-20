"""
G8.4 - inference_pipeline_finetune.py
Fine-tuned Xception + Bi-LSTM - Celeb-DF v2

- Giữ nguyên preprocessing: 15 frame, MTCNN 224x224, margin=20.
- Threshold phân loại: 0.98.
- Hỗ trợ visualization 15 frame với bounding box.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(r"D:\Project")
SRC_DIR = PROJECT_DIR / "src"

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints_celebdf_finetune"
    / "best_model.pth"
)

SEQUENCE_LENGTH = 15
IMAGE_SIZE = 224
MARGIN = 20
THRESHOLD = 0.98

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(1, 3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(1, 3, 1, 1)


class VideoFacePreprocessor:
    """Video -> 15 face frames + metadata + visualization."""

    SUPPORTED_EXTENSIONS = {
        ".mp4", ".avi", ".mov", ".mkv", ".webm"
    }

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        image_size: int = IMAGE_SIZE,
        margin: int = MARGIN,
        device: Optional[torch.device] = None,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length phải > 0.")
        if image_size <= 0:
            raise ValueError("image_size phải > 0.")
        if margin < 0:
            raise ValueError("margin không được âm.")

        self.sequence_length = sequence_length
        self.image_size = image_size
        self.margin = margin
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"[INFO] MTCNN device: {self.device}")

        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            keep_all=False,
            select_largest=True,
            post_process=False,
            device=self.device,
        )

    @staticmethod
    def face_tensor_to_uint8(face: torch.Tensor) -> np.ndarray:
        """(3,H,W) Tensor -> (H,W,3) uint8."""
        if not isinstance(face, torch.Tensor):
            raise TypeError(
                f"MTCNN output phải là Tensor, nhận {type(face)}."
            )
        if face.ndim != 3 or face.shape[0] != 3:
            raise ValueError(
                f"MTCNN output shape không hợp lệ: {tuple(face.shape)}."
            )

        face_np = (
            face.permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        return np.clip(face_np, 0, 255).astype(np.uint8)

    @staticmethod
    def sanitize_box(
        box: Optional[np.ndarray],
        width: int,
        height: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Đưa box vào phạm vi ảnh."""
        if box is None:
            return None
        try:
            x1, y1, x2, y2 = [
                int(round(float(v))) for v in box
            ]
        except (TypeError, ValueError):
            return None

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

    @classmethod
    def draw_box(
        cls,
        frame_rgb: np.ndarray,
        box: Optional[np.ndarray],
        confidence: Optional[float],
        text_when_missing: str = "No face detected",
    ) -> np.ndarray:
        """Vẽ bounding box lên bản sao frame."""
        image = frame_rgb.copy()
        h, w = image.shape[:2]

        safe = cls.sanitize_box(box, w, h)
        if safe is None:
            cv2.putText(
                image,
                text_when_missing,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 165, 0),
                2,
                cv2.LINE_AA,
            )
            return image

        x1, y1, x2, y2 = safe
        green = (0, 255, 0)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            green,
            3,
        )

        label = (
            f"Face: {float(confidence):.2f}"
            if confidence is not None
            else "Face detected"
        )

        cv2.putText(
            image,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            green,
            2,
            cv2.LINE_AA,
        )
        return image

    def detect_box(
        self,
        image_pil: Image.Image,
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Lấy box/confidence cho visualization."""
        try:
            with torch.no_grad():
                boxes, probs = self.mtcnn.detect(image_pil)

            if boxes is None or probs is None or len(boxes) == 0:
                return None, None

            return (
                np.asarray(boxes[0], dtype=np.float32),
                float(probs[0]),
            )
        except Exception as exc:
            print(f"[WARN] detect box thất bại: {exc}")
            return None, None

    def _process(
        self,
        video_path: str | Path,
        visualize: bool,
    ) -> Tuple[
        np.ndarray,
        Dict[str, Any],
        List[Dict[str, Any]],
    ]:
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy video:\n{video_path}"
            )
        if not video_path.is_file():
            raise ValueError(
                f"Đường dẫn không phải file:\n{video_path}"
            )
        if video_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Định dạng không được hỗ trợ: {video_path.suffix}"
            )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(
                f"Không thể mở video:\n{video_path}"
            )

        try:
            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )
            if total_frames < self.sequence_length:
                raise ValueError(
                    f"Video có {total_frames} frame; "
                    f"cần ít nhất {self.sequence_length} frame."
                )

            frame_indices = np.linspace(
                0,
                total_frames - 1,
                self.sequence_length,
                dtype=int,
            )

            face_sequence: List[np.ndarray] = []
            visual_frames: List[Dict[str, Any]] = []

            detected_count = 0
            fallback_count = 0
            read_fail_count = 0
            padding_count = 0

            last_valid_face: Optional[np.ndarray] = None

            for pos, frame_index in enumerate(
                frame_indices, start=1
            ):
                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    int(frame_index),
                )
                ret, frame_bgr = cap.read()

                if not ret or frame_bgr is None:
                    read_fail_count += 1
                    if visualize:
                        visual_frames.append({
                            "sequence_position": pos,
                            "frame_index": int(frame_index),
                            "image": None,
                            "detected": False,
                            "fallback": False,
                            "read_error": True,
                            "face_probability": None,
                        })
                    continue

                frame_rgb = cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2RGB,
                )
                image_pil = Image.fromarray(frame_rgb)

                # Đây là crop dùng cho model.
                with torch.no_grad():
                    face = self.mtcnn(image_pil)

                box = None
                face_probability = None
                if visualize:
                    box, face_probability = self.detect_box(
                        image_pil
                    )

                if face is not None:
                    face_np = self.face_tensor_to_uint8(face)
                    face_sequence.append(face_np)
                    last_valid_face = face_np.copy()
                    detected_count += 1

                    if visualize:
                        display = self.draw_box(
                            frame_rgb,
                            box,
                            face_probability,
                            "Face detected",
                        )
                        visual_frames.append({
                            "sequence_position": pos,
                            "frame_index": int(frame_index),
                            "image": display,
                            "detected": True,
                            "fallback": False,
                            "read_error": False,
                            "face_probability": face_probability,
                        })

                elif last_valid_face is not None:
                    face_sequence.append(last_valid_face.copy())
                    fallback_count += 1

                    if visualize:
                        display = frame_rgb.copy()
                        cv2.putText(
                            display,
                            "Fallback face",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.85,
                            (255, 165, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        visual_frames.append({
                            "sequence_position": pos,
                            "frame_index": int(frame_index),
                            "image": display,
                            "detected": False,
                            "fallback": True,
                            "read_error": False,
                            "face_probability": None,
                        })

                elif visualize:
                    display = self.draw_box(
                        frame_rgb,
                        None,
                        None,
                    )
                    visual_frames.append({
                        "sequence_position": pos,
                        "frame_index": int(frame_index),
                        "image": display,
                        "detected": False,
                        "fallback": False,
                        "read_error": False,
                        "face_probability": None,
                    })

            if not face_sequence:
                raise RuntimeError(
                    "Không phát hiện được khuôn mặt ở 15 frame mẫu."
                )

            while len(face_sequence) < self.sequence_length:
                face_sequence.append(face_sequence[-1].copy())
                padding_count += 1

            sequence = np.asarray(
                face_sequence[:self.sequence_length],
                dtype=np.uint8,
            )

            expected = (
                self.sequence_length,
                self.image_size,
                self.image_size,
                3,
            )
            if sequence.shape != expected:
                raise RuntimeError(
                    f"Sequence shape {sequence.shape}; "
                    f"kỳ vọng {expected}."
                )

            if visualize:
                visual_frames = visual_frames[
                    :self.sequence_length
                ]
                while len(visual_frames) < self.sequence_length:
                    visual_frames.append({
                        "sequence_position": len(visual_frames) + 1,
                        "frame_index": None,
                        "image": None,
                        "detected": False,
                        "fallback": False,
                        "read_error": True,
                        "face_probability": None,
                    })

            metadata: Dict[str, Any] = {
                "video_path": str(video_path),
                "total_frames": total_frames,
                "sampled_frame_indices": frame_indices.tolist(),
                "detected_faces": detected_count,
                "fallback_faces": fallback_count,
                "padding_faces": padding_count,
                "read_failures": read_fail_count,
                "sequence_shape": tuple(sequence.shape),
            }

            return sequence, metadata, visual_frames

        finally:
            cap.release()

    def process_video(
        self,
        video_path: str | Path,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """API cũ, giữ tương thích."""
        sequence, metadata, _ = self._process(
            video_path,
            visualize=False,
        )
        return sequence, metadata

    def process_video_with_visualization(
        self,
        video_path: str | Path,
    ) -> Tuple[
        np.ndarray,
        Dict[str, Any],
        List[Dict[str, Any]],
    ]:
        """API cho Streamlit."""
        return self._process(
            video_path,
            visualize=True,
        )

    def npy_to_tensor(
        self,
        sequence: np.ndarray,
    ) -> torch.Tensor:
        """(15,224,224,3) uint8 -> (1,15,3,224,224) float32."""
        expected = (
            self.sequence_length,
            self.image_size,
            self.image_size,
            3,
        )
        if sequence.shape != expected:
            raise ValueError(
                f"Shape không đúng: {sequence.shape}; "
                f"kỳ vọng {expected}."
            )
        if sequence.dtype != np.uint8:
            raise ValueError(
                f"dtype phải uint8, nhận {sequence.dtype}."
            )

        tensor = torch.from_numpy(
            sequence
        ).float() / 255.0

        tensor = tensor.permute(
            0, 3, 1, 2
        ).contiguous()

        tensor = (
            tensor - IMAGENET_MEAN
        ) / IMAGENET_STD

        return tensor.unsqueeze(0)


class FineTunePredictor:
    """
    Predictor cho checkpoint fine-tuned.
    Constructor nhận checkpoint_path, threshold, device.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = CHECKPOINT_PATH,
        threshold: float = THRESHOLD,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.threshold = float(threshold)

        if not 0.0 < self.threshold < 1.0:
            raise ValueError(
                "threshold phải nằm trong khoảng (0,1)."
            )

        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint:\n{self.checkpoint_path}"
            )

        if not SRC_DIR.exists():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục model:\n{SRC_DIR}"
            )

        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))

        try:
            from model_finetune import (
                SpatialTemporalXceptionBiLSTM_FineTune
            )
        except Exception as exc:
            raise ImportError(
                "Không import được "
                "SpatialTemporalXceptionBiLSTM_FineTune "
                "từ D:\\Project\\src\\model_finetune.py."
            ) from exc

        self.preprocessor = VideoFacePreprocessor(
            sequence_length=SEQUENCE_LENGTH,
            image_size=IMAGE_SIZE,
            margin=MARGIN,
            device=self.device,
        )

        self.model = SpatialTemporalXceptionBiLSTM_FineTune(
            sequence_length=SEQUENCE_LENGTH,
            lstm_hidden_size=256,
            lstm_layers=2,
            dropout=0.5,
        )

        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
            )
        except Exception as exc:
            raise RuntimeError(
                "Không thể đọc checkpoint."
            ) from exc

        if not isinstance(checkpoint, dict):
            raise ValueError(
                "Checkpoint không phải dictionary."
            )

        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint),
        )

        if not isinstance(state_dict, dict):
            raise ValueError(
                "state_dict không hợp lệ."
            )

        cleaned = {}
        for key, value in state_dict.items():
            cleaned[
                key[7:] if key.startswith("module.") else key
            ] = value

        try:
            self.model.load_state_dict(
                cleaned,
                strict=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Checkpoint không khớp kiến trúc model_finetune.py."
            ) from exc

        self.model = self.model.to(self.device)
        self.model.eval()

        self.checkpoint_epoch = checkpoint.get("epoch")
        self.checkpoint_val_auc = None

        val_metrics = checkpoint.get("val_metrics")
        if isinstance(val_metrics, dict):
            auc = val_metrics.get("roc_auc")
            if auc is not None:
                self.checkpoint_val_auc = float(auc)

        if self.checkpoint_val_auc is None:
            auc = checkpoint.get("best_val_auc")
            if auc is not None:
                self.checkpoint_val_auc = float(auc)

        print(f"[INFO] Model device: {self.device}")
        print(f"[INFO] Threshold: {self.threshold:.2f}")

    @torch.no_grad()
    def predict_sequence(
        self,
        sequence: np.ndarray,
    ) -> Dict[str, Any]:
        """Dự đoán từ face sequence."""
        tensor = (
            self.preprocessor
            .npy_to_tensor(sequence)
            .to(self.device, non_blocking=True)
        )

        logits = self.model(tensor)
        logit_tensor = logits.reshape(-1)[0]

        fake_probability = float(
            torch.sigmoid(logit_tensor).cpu().item()
        )
        fake_probability = min(
            max(fake_probability, 0.0),
            1.0,
        )

        real_probability = 1.0 - fake_probability
        predicted_label = int(
            fake_probability >= self.threshold
        )

        return {
            "logit": float(logit_tensor.cpu().item()),
            "fake_probability": fake_probability,
            "fake_percentage": fake_probability * 100.0,
            "real_probability": real_probability,
            "real_percentage": real_probability * 100.0,
            "predicted_label": predicted_label,
            "predicted_class": (
                "FAKE" if predicted_label == 1 else "REAL"
            ),
            "threshold": self.threshold,
        }

    @torch.no_grad()
    def predict_video(
        self,
        video_path: str | Path,
    ) -> Dict[str, Any]:
        """Video -> prediction."""
        sequence, metadata = (
            self.preprocessor.process_video(video_path)
        )
        result = self.predict_sequence(sequence)
        result.update(metadata)
        return result

    @torch.no_grad()
    def predict_video_with_visualization(
        self,
        video_path: str | Path,
    ) -> Dict[str, Any]:
        """Video -> prediction + 15 visualized frames."""
        (
            sequence,
            metadata,
            visual_frames,
        ) = self.preprocessor.process_video_with_visualization(
            video_path
        )

        result = self.predict_sequence(sequence)
        result.update(metadata)
        result["visual_frames"] = visual_frames
        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference Fine-tuned Xception + Bi-LSTM"
    )
    parser.add_argument(
        "video",
        type=str,
        help="Đường dẫn video.",
    )
    args = parser.parse_args()

    predictor = FineTunePredictor()
    result = predictor.predict_video(args.video)

    print("\n" + "=" * 70)
    print("INFERENCE RESULT")
    print("=" * 70)
    print(f"Video            : {result['video_path']}")
    print(f"Total frames     : {result['total_frames']}")
    print(f"Sampled frames   : {SEQUENCE_LENGTH}")
    print(f"Detected faces   : {result['detected_faces']}")
    print(f"Fallback faces   : {result['fallback_faces']}")
    print(f"Padding faces    : {result['padding_faces']}")
    print(f"Read failures    : {result['read_failures']}")
    print(
        f"Fake probability : "
        f"{result['fake_percentage']:.4f}%"
    )
    print(
        f"Real probability : "
        f"{result['real_percentage']:.4f}%"
    )
    print(f"Threshold        : {result['threshold']:.2f}")
    print(f"Prediction       : {result['predicted_class']}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Người dùng dừng chương trình.")
        sys.exit(1)
    except Exception as exc:
        print("\n[ERROR] Inference thất bại.")
        print(f"[ERROR] {exc}")
        raise
