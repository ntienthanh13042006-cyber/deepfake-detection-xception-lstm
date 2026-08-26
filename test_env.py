import torch
import cv2
import numpy as np
from facenet_pytorch import MTCNN

print("=== KIỂM TRA MÔI TRƯỜNG ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA (GPU) khả dụng: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
print(f"OpenCV version: {cv2.__version__}")
print(" Môi trường đã sẵn sàng cho dự án!")