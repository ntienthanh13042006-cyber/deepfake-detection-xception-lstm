"""
Mô hình Xception + Bi-LSTM có Fine-tuning một phần Xception.

Chiến lược:
- Xception pretrained ImageNet.
- Freeze block1 -> block11.
- Fine-tune block12 + conv3 + conv4.
- Giữ BatchNorm ở chế độ eval để ổn định running statistics
  với batch size nhỏ = 2.
- Bi-LSTM và classifier được train bình thường.

Kiến trúc:
Input (B, 15, 3, 224, 224)
    ↓
Xception
    ↓
2048-D/frame
    ↓
Bi-LSTM
    ↓
512-D/frame
    ↓
Mean temporal pooling
    ↓
512-D
    ↓
FC 512 → 128
    ↓
ReLU + Dropout
    ↓
FC 128 → 1
    ↓
Logit
"""

import sys

import torch
import torch.nn as nn
import timm


class SpatialTemporalXceptionBiLSTM_FineTune(nn.Module):
    """
    Xception + Bi-LSTM với fine-tuning phần cuối Xception.
    """

    def __init__(
        self,
        sequence_length=15,
        lstm_hidden_size=256,
        lstm_layers=2,
        dropout=0.5
    ):
        super().__init__()

        self.sequence_length = sequence_length

        # ============================================================
        # 1. Xception pretrained ImageNet
        # ============================================================
        self.xception = timm.create_model(
            "legacy_xception",
            pretrained=True,
            num_classes=0
        )

        self.feature_dim = 2048

        # ============================================================
        # 2. Freeze toàn bộ Xception trước
        # ============================================================
        for param in self.xception.parameters():
            param.requires_grad = False

        # ============================================================
        # 3. Fine-tune phần cuối Xception
        #
        # block12:
        #   block cuối của Xception trước conv3/conv4
        #
        # conv3 + conv4:
        #   chuyển feature cuối lên 2048-D
        # ============================================================

        for param in self.xception.block12.parameters():
            param.requires_grad = True

        for param in self.xception.conv3.parameters():
            param.requires_grad = True

        for param in self.xception.bn3.parameters():
            param.requires_grad = True

        for param in self.xception.conv4.parameters():
            param.requires_grad = True

        for param in self.xception.bn4.parameters():
            param.requires_grad = True

        # ============================================================
        # 4. Bi-LSTM
        # ============================================================
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # ============================================================
        # 5. Classifier
        # ============================================================
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def _freeze_batchnorm_statistics(self):
        """
        Giữ BatchNorm ở eval để không cập nhật running mean/variance.

        Batch size của dự án chỉ = 2 nên việc này giúp ổn định
        quá trình fine-tuning.
        """

        for module in self.xception.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def train(self, mode=True):
        """
        Override train():

        - Cho toàn bộ model vào train/eval bình thường.
        - Sau đó ép tất cả BatchNorm của Xception về eval.
        """

        super().train(mode)

        if mode:
            self._freeze_batchnorm_statistics()

        return self

    def forward(self, x):
        """
        Input:
            x: (B, T, 3, 224, 224)

        Output:
            logits: (B, 1)
        """

        if x.ndim != 5:
            raise ValueError(
                f"Input phải có 5 chiều (B,T,C,H,W), "
                f"nhưng nhận được shape={tuple(x.shape)}"
            )

        batch_size, seq_len, channels, height, width = x.shape

        if seq_len != self.sequence_length:
            raise ValueError(
                f"sequence_length phải bằng {self.sequence_length}, "
                f"nhưng nhận được {seq_len}"
            )

        if channels != 3:
            raise ValueError(
                f"Input phải có 3 channel RGB, "
                f"nhưng nhận được {channels}"
            )

        # ============================================================
        # 1. Gộp batch + thời gian
        # ============================================================
        x = x.reshape(
            batch_size * seq_len,
            channels,
            height,
            width
        )

        # ============================================================
        # 2. Xception
        # ============================================================
        spatial_features = self.xception(x)

        # Kỳ vọng:
        # (B*T, 2048)

        if spatial_features.ndim != 2:
            raise RuntimeError(
                "Xception không trả về tensor 2 chiều. "
                f"Shape nhận được: {tuple(spatial_features.shape)}"
            )

        if spatial_features.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"Số chiều feature không đúng. "
                f"Kỳ vọng {self.feature_dim}, "
                f"nhận {spatial_features.shape[1]}"
            )

        # ============================================================
        # 3. Khôi phục chiều thời gian
        # ============================================================
        temporal_features = spatial_features.reshape(
            batch_size,
            seq_len,
            self.feature_dim
        )

        # Shape:
        # (B, T, 2048)

        # ============================================================
        # 4. Bi-LSTM
        # ============================================================
        lstm_output, _ = self.lstm(temporal_features)

        # Shape:
        # (B, T, 512)

        # ============================================================
        # 5. Mean pooling theo thời gian
        # ============================================================
        temporal_feature = lstm_output.mean(dim=1)

        # Shape:
        # (B, 512)

        # ============================================================
        # 6. Classifier
        # ============================================================
        logits = self.classifier(temporal_feature)

        # Shape:
        # (B, 1)

        return logits


def print_trainable_parameters(model):
    """
    In thống kê số parameter trainable/frozen.
    """

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen = total - trainable

    print("\n" + "=" * 80)
    print("THỐNG KÊ PARAMETER")
    print("=" * 80)

    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"Frozen parameters    : {frozen:,}")

    print(
        f"Trainable ratio      : "
        f"{trainable / total * 100:.2f}%"
    )

    print("=" * 80)


def main():
    """
    Smoke test:
    - Khởi tạo model.
    - Kiểm tra trainable parameters.
    - Test forward bằng tensor giả.
    """

    try:
        print("=" * 80)
        print("SMOKE TEST - XCEPTION + BI-LSTM FINE-TUNING")
        print("=" * 80)

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"[INFO] Device: {device}")

        if torch.cuda.is_available():
            print(
                f"[INFO] GPU: "
                f"{torch.cuda.get_device_name(0)}"
            )

        # ------------------------------------------------------------
        # Khởi tạo model
        # ------------------------------------------------------------
        print("\n[INFO] Khởi tạo model...")

        model = SpatialTemporalXceptionBiLSTM_FineTune(
            sequence_length=15,
            lstm_hidden_size=256,
            lstm_layers=2,
            dropout=0.5
        )

        model = model.to(device)

        # ------------------------------------------------------------
        # In parameter
        # ------------------------------------------------------------
        print_trainable_parameters(model)

        # ------------------------------------------------------------
        # Kiểm tra trạng thái các phần Xception
        # ------------------------------------------------------------
        print("\n" + "=" * 80)
        print("KIỂM TRA TRẠNG THÁI FINE-TUNING")
        print("=" * 80)

        print(
            f"block11 trainable: "
            f"{any(p.requires_grad for p in model.xception.block11.parameters())}"
        )

        print(
            f"block12 trainable: "
            f"{any(p.requires_grad for p in model.xception.block12.parameters())}"
        )

        print(
            f"conv3 trainable: "
            f"{any(p.requires_grad for p in model.xception.conv3.parameters())}"
        )

        print(
            f"conv4 trainable: "
            f"{any(p.requires_grad for p in model.xception.conv4.parameters())}"
        )

        # ------------------------------------------------------------
        # Kiểm tra forward
        # ------------------------------------------------------------
        print("\n[INFO] Kiểm tra forward...")

        model.eval()

        dummy_input = torch.randn(
            2,
            15,
            3,
            224,
            224,
            device=device
        )

        with torch.no_grad():
            output = model(dummy_input)

        print(
            f"[INFO] Input shape : "
            f"{tuple(dummy_input.shape)}"
        )

        print(
            f"[INFO] Output shape: "
            f"{tuple(output.shape)}"
        )

        if tuple(output.shape) != (2, 1):
            raise RuntimeError(
                f"Output shape không đúng: "
                f"{tuple(output.shape)}"
            )

        print("\n[PASS] Smoke test thành công.")

    except Exception as e:
        print("\n[ERROR] Smoke test thất bại.")
        print(f"[ERROR] Chi tiết: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()