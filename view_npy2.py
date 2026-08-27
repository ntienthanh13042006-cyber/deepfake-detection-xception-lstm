import numpy as np
import matplotlib.pyplot as plt

# Điền trực tiếp đường dẫn tới file .npy bất kỳ bạn muốn mở
npy_path = r"D:\Project\Processed_Data\Fake\Deepfakes\008_990.npy"  # Thay tên file tại đây

# Đọc và hiển thị
faces = np.load(npy_path)
print(f"[INFO] Kích thước dữ liệu: {faces.shape}")

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
fig.suptitle(f"File: {npy_path}", fontsize=12)

for i in range(min(15, len(faces))):
    row, col = i // 5, i % 5
    axes[row, col].imshow(faces[i])
    axes[row, col].set_title(f"Frame {i+1}")
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()