import os
import numpy as np
import matplotlib.pyplot as plt

# 1. Chỉ định thư mục chứa dữ liệu .npy
folder_path = r"D:\Project\Processed_Data\Real"
# Nếu muốn xem file Fake, đổi thành: r"D:\Project\Processed_Data\Fake\Deepfakes"

# 2. Kiểm tra thư mục và tự động lấy file .npy đầu tiên
if not os.path.exists(folder_path):
    print(f"[LỖI] Không tìm thấy thư mục: {folder_path}")
else:
    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    
    if len(npy_files) == 0:
        print(f"[LỖI] Thư mục {folder_path} trống, chưa có file .npy nào!")
    else:
        # Lấy file đầu tiên trong danh sách
        sample_file = npy_files[0]
        npy_path = os.path.join(folder_path, sample_file)
        
        print(f"[INFO] Tự động chọn file: {sample_file}")
        
        # Đọc dữ liệu
        faces = np.load(npy_path)
        print(f"[INFO] Kích thước dữ liệu (Shape): {faces.shape}")

        # 3. Hiển thị 15 khung hình khuôn mặt
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        fig.suptitle(f"Tệp: {sample_file} | Cấu trúc: {faces.shape}", fontsize=14, fontweight='bold')

        for i in range(min(15, len(faces))):
            row = i // 5
            col = i % 5
            axes[row, col].imshow(faces[i])
            axes[row, col].set_title(f"Frame {i+1}")
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()