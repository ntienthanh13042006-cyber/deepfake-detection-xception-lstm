import torch
import torch.nn as nn
import timm

class SpatialTemporalXceptionBiLSTM(nn.Module):
    def __init__(self, sequence_length=15, lstm_hidden_size=256, lstm_layers=2):
        super(SpatialTemporalXceptionBiLSTM, self).__init__()
        
        self.seq_len = sequence_length
        
        # -------------------------------------------------------------
        # 1. SPATIAL FEATURE EXTRACTOR: Xception (Pre-trained ImageNet)
        # -------------------------------------------------------------
        # Tải mô hình Xception, bỏ lớp Fully Connected phân loại cuối (num_classes=0)
        self.spatial_extractor = timm.create_model('legacy_xception', pretrained=True, num_classes=0)
        
        # Đóng băng trọng số Xception (Freeze Weights) để chống overfitting và tăng tốc huấn luyện
        for param in self.spatial_extractor.parameters():
            param.requires_grad = False
            
        feature_dim = self.spatial_extractor.num_features # 2048 chiều
        
        # -------------------------------------------------------------
        # 2. TEMPORAL FEATURE PROCESSOR: Bi-LSTM
        # -------------------------------------------------------------
        self.bi_lstm = nn.LSTM(
            input_size=feature_dim,     # Đầu vào: 2048
            hidden_size=lstm_hidden_size,# Kích thước hidden state: 256
            num_layers=lstm_layers,     # Số lớp LSTM xếp chồng: 2
            batch_first=True,
            bidirectional=True,         # Hai chiều (Forward + Backward)
            dropout=0.5                 # Anti-overfitting
        )
        
        # -------------------------------------------------------------
        # 3. CLASSIFIER HEAD: Fully Connected Layers
        # -------------------------------------------------------------
        # Bi-LSTM 2 chiều nên kích thước đầu ra là hidden_size * 2 = 512
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1) # Đưa ra Logit phục vụ hàm Binary Cross-Entropy Loss
        )

    def forward(self, x):
        """
        :param x: Input Tensor shape (Batch_Size, Sequence_Length, Channels, Height, Width)
                  Ví dụ: (B, 15, 3, 224, 224)
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Bước 1: Gộp Batch và Sequence để đưa vào mạng CNN trích xuất cùng lúc
        # Shape: (B * 15, 3, 224, 224)
        x_reshaped = x.view(batch_size * seq_len, c, h, w)
        
        # Trích xuất đặc trưng không gian qua Xception
        # Shape: (B * 15, 2048)
        spatial_features = self.spatial_extractor(x_reshaped)
        
        # Bước 2: Tách lại về dạng chuỗi thời gian cho Bi-LSTM
        # Shape: (B, 15, 2048)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)
        
        # Bước 3: Truyền qua Bi-LSTM
        # lstm_out shape: (B, 15, 512)
        lstm_out, _ = self.bi_lstm(spatial_features)
        
        # Bước 4: Temporal Pooling (Lấy trung bình đặc trưng của 15 khung hình)
        # Shape: (B, 512)
        aggregated_features = torch.mean(lstm_out, dim=1)
        
        # Bước 5: Phân loại Real/Fake
        # Logits shape: (B, 1)
        logits = self.classifier(aggregated_features)
        
        return logits

# ----------------- CODE KIỂM TRA LUỒNG DỮ LIỆU (TEST RUN) -----------------
if __name__ == '__main__':
    print("[INFO] Khởi tạo mô hình Xception + Bi-LSTM...")
    model = SpatialTemporalXceptionBiLSTM(sequence_length=15)
    model.eval()
    
    # Giả lập 1 Batch dữ liệu gồm 2 video: (Batch_Size=2, Seq_Len=15, C=3, H=224, W=224)
    dummy_input = torch.randn(2, 15, 3, 224, 224)
    
    print(f"[INFO] Kích thước Tensor đầu vào: {dummy_input.shape}")
    
    with torch.no_grad():
        output_logits = model(dummy_input)
        
    print(f"[INFO] Kích thước Output Logits: {output_logits.shape}") # (2, 1)
    print(" Kiểm tra luồng dữ liệu kiến trúc mô hình thành công!")