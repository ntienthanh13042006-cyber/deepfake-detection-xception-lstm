import torch
import torch.nn as nn
import timm


class SpatialTemporalXceptionBiLSTM(nn.Module):
    """
    Mô hình phát hiện Deepfake kết hợp:

        Xception
            ↓
        Spatial Feature
            ↓
        Bi-LSTM
            ↓
        Temporal Feature
            ↓
        Fully Connected Classifier
            ↓
        Real / Fake Logit
    """

    def __init__(
        self,
        sequence_length=15,
        lstm_hidden_size=256,
        lstm_layers=2
    ):

        super().__init__()

        self.seq_len = sequence_length

        # =====================================================
        # 1. XCEPTION
        # =====================================================
        self.spatial_extractor = timm.create_model(
            "legacy_xception",
            pretrained=True,
            num_classes=0
        )

        # -----------------------------------------------------
        # Freeze toàn bộ Xception
        # -----------------------------------------------------
        for param in self.spatial_extractor.parameters():
            param.requires_grad = False

        # -----------------------------------------------------
        # Số chiều feature đầu ra
        # -----------------------------------------------------
        feature_dim = self.spatial_extractor.num_features

        print(
            f"[MODEL] Xception feature dimension: "
            f"{feature_dim}"
        )

        # =====================================================
        # 2. BI-LSTM
        # =====================================================
        self.bi_lstm = nn.LSTM(

            input_size=feature_dim,

            hidden_size=lstm_hidden_size,

            num_layers=lstm_layers,

            batch_first=True,

            bidirectional=True,

            dropout=0.5 if lstm_layers > 1 else 0.0
        )

        # =====================================================
        # 3. CLASSIFIER
        # =====================================================
        self.classifier = nn.Sequential(

            nn.Linear(
                lstm_hidden_size * 2,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                128,
                1
            )
        )

        # -----------------------------------------------------
        # Đảm bảo backbone ở eval mode ngay từ đầu
        # -----------------------------------------------------
        self.spatial_extractor.eval()

    # =========================================================
    # OVERRIDE TRAIN
    # =========================================================
    def train(self, mode=True):

        # Cho toàn bộ model vào train/eval
        super().train(mode)

        # -----------------------------------------------------
        # QUAN TRỌNG:
        # Xception vẫn luôn ở eval mode
        # -----------------------------------------------------
        self.spatial_extractor.eval()

        return self

    # =========================================================
    # FORWARD
    # =========================================================
    def forward(self, x):

        # -----------------------------------------------------
        # Kiểm tra shape đầu vào
        # -----------------------------------------------------
        if x.ndim != 5:

            raise ValueError(
                "Input phải có 5 chiều: "
                "(B, T, C, H, W)"
            )

        batch_size, seq_len, channels, height, width = x.shape

        # -----------------------------------------------------
        # Kiểm tra sequence length
        # -----------------------------------------------------
        if seq_len != self.seq_len:

            raise ValueError(
                f"Sequence length không đúng. "
                f"Model yêu cầu {self.seq_len}, "
                f"nhưng nhận {seq_len}."
            )

        # =====================================================
        # STEP 1: CNN
        # =====================================================

        # (B,T,C,H,W)
        #      ↓
        # (B*T,C,H,W)
        x = x.reshape(
            batch_size * seq_len,
            channels,
            height,
            width
        )

        # -----------------------------------------------------
        # Xception frozen:
        # Không cần tạo computational graph
        # -----------------------------------------------------
        with torch.no_grad():

            spatial_features = (
                self.spatial_extractor(x)
            )

        # Shape:
        # (B*T, 2048)

        # =====================================================
        # STEP 2: RESTORE TEMPORAL DIMENSION
        # =====================================================

        spatial_features = spatial_features.reshape(

            batch_size,
            seq_len,
            -1
        )

        # Shape:
        # (B,T,2048)

        # =====================================================
        # STEP 3: BI-LSTM
        # =====================================================

        lstm_out, _ = self.bi_lstm(
            spatial_features
        )

        # Shape:
        # (B,T,512)

        # =====================================================
        # STEP 4: TEMPORAL POOLING
        # =====================================================

        aggregated_features = torch.mean(
            lstm_out,
            dim=1
        )

        # Shape:
        # (B,512)

        # =====================================================
        # STEP 5: CLASSIFIER
        # =====================================================

        logits = self.classifier(
            aggregated_features
        )

        # Shape:
        # (B,1)

        return logits


# =============================================================
# TEST MODEL
# =============================================================
if __name__ == "__main__":

    print(
        "\n[INFO] Khởi tạo Xception + Bi-LSTM..."
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[INFO] Device: {device}"
    )

    model = SpatialTemporalXceptionBiLSTM(
        sequence_length=15,
        lstm_hidden_size=256,
        lstm_layers=2
    )

    model = model.to(device)

    # ---------------------------------------------------------
    # Kiểm tra số parameter
    # ---------------------------------------------------------
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"[INFO] Tổng parameters: "
        f"{total_params:,}"
    )

    print(
        f"[INFO] Trainable parameters: "
        f"{trainable_params:,}"
    )

    # ---------------------------------------------------------
    # Dummy input
    # ---------------------------------------------------------
    dummy_input = torch.randn(
        2,
        15,
        3,
        224,
        224,
        device=device
    )

    print(
        f"[INFO] Input shape: "
        f"{dummy_input.shape}"
    )

    # ---------------------------------------------------------
    # Forward test
    # ---------------------------------------------------------
    model.eval()

    with torch.no_grad():

        output = model(
            dummy_input
        )

    print(
        f"[INFO] Output shape: "
        f"{output.shape}"
    )

    print(
        "[INFO] Kiểm tra mô hình thành công!"
    )