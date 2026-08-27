import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir):
        """
        Custom Dataset để nạp các file .npy tiền xử lý.
        :param data_dir: Thư mục gốc chứa dữ liệu (D:\\Project\\Processed_Data)
        """
        self.file_paths = []
        self.labels = []
        
        # 1. Gom các file Real (Nhãn 0)
        real_dir = os.path.join(data_dir, "Real")
        if os.path.exists(real_dir):
            for fname in os.listdir(real_dir):
                if fname.endswith('.npy'):
                    self.file_paths.append(os.path.join(real_dir, fname))
                    self.labels.append(0) # 0 đại diện cho REAL
                    
        # 2. Gom các file Fake (Nhãn 1) từ tất cả các thư mục con
        fake_base_dir = os.path.join(data_dir, "Fake")
        if os.path.exists(fake_base_dir):
            for root, _, files in os.walk(fake_base_dir):
                for fname in files:
                    if fname.endswith('.npy'):
                        self.file_paths.append(os.path.join(root, fname))
                        self.labels.append(1) # 1 đại diện cho FAKE

        print(f"[DATASET] Tổng số mẫu nạp thành công: {len(self.file_paths)} "
              f"(Real: {self.labels.count(0)}, Fake: {self.labels.count(1)})")

        # Transform chuẩn hóa ảnh khuôn mặt theo chuẩn ImageNet
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. Nạp file .npy shape (15, 224, 224, 3)
        np_data = np.load(self.file_paths[idx])
        
        # 2. Chuyển đổi kiểu dữ liệu về FloatTensor và scale giá trị pixel [0, 1]
        tensor_data = torch.from_numpy(np_data).float() / 255.0
        
        # 3. Đổi trục từ (N, H, W, C) sang (N, C, H, W) chuẩn PyTorch
        tensor_data = tensor_data.permute(0, 3, 1, 2)
        
        # 4. Chuẩn hóa từng frame trong chuỗi 15 frames
        for i in range(tensor_data.size(0)):
            tensor_data[i] = self.transform(tensor_data[i])

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return tensor_data, label
    
  # --- ĐOẠN CODE KIỂM THỬ (TEST BLOCK) ĐÃ SỬA LỖI ---
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # 1. Khai báo đường dẫn (Sử dụng raw string 'r' để tránh lỗi \P)
    data_dir = r"D:\Project\Processed_Data"
    
    print("Đang khởi tạo Dataset...")
    
    # Đã xóa bỏ tham số sequence_length=15 gây lỗi
    dataset = DeepfakeDataset(data_dir=data_dir)
    
    print(f"✅ Tổng số video đã nhận diện được: {len(dataset)}")
    
    # 2. Kiểm tra việc lấy mẫu dữ liệu
    if len(dataset) > 0:
        frames, label = dataset[0]
        print("\n--- Kiểm tra 1 mẫu dữ liệu đầu tiên ---")
        print(f"Kích thước Tensor (Frames): {frames.shape}") # Kỳ vọng: (15, 3, 224, 224) hoặc (3, 15, 224, 224)
        print(f"Kiểu dữ liệu: {frames.dtype}")
        print(f"Nhãn (Label - 0 là Real, 1 là Fake): {label}")
        
        # 3. Kiểm tra thông qua DataLoader (Gói thành Batch)
        print("\n--- Kiểm tra DataLoader ---")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        batch_frames, batch_labels = next(iter(dataloader))
        print(f"Kích thước Batch Tensor: {batch_frames.shape}")
        print(f"Batch Labels: {batch_labels}")
        print("✅ Giai đoạn 2 (Dataset) đã hoạt động hoàn hảo!")
    else:
        # Đã sửa lỗi \P bằng cách dùng dấu gạch chéo kép \\
        print("❌ Không tìm thấy dữ liệu. Hãy kiểm tra lại thư mục D:\\Project\\Processed_Data")