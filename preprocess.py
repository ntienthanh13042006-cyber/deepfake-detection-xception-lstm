import os
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

class VideoPreprocessor:
    def __init__(self, sequence_length=15, image_size=224, margin=20):
        """
        Khởi tạo pipeline tiền xử lý video.
        :param sequence_length: Số lượng khung hình (N) trích xuất mỗi video.
        :param image_size: Kích thước khuôn mặt đầu ra (224x224 cho Xception).
        :param margin: Lề mở rộng khi crop khuôn mặt (pixel).
        """
        self.seq_len = sequence_length
        self.img_size = image_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"[INFO] Sử dụng thiết bị: {self.device}")
        
        # MTCNN tự động resize về image_size và có hỗ trợ margin
        self.mtcnn = MTCNN(
            image_size=self.img_size, 
            margin=margin, 
            keep_all=False, 
            select_largest=True, 
            post_process=False, 
            device=self.device
        )

    def process_video(self, video_path):
        """
        Trích xuất chuỗi khuôn mặt từ một video.
        :return: Numpy array shape (N, 224, 224, 3) hoặc None nếu lỗi.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Không thể đọc video: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < self.seq_len:
                raise ValueError("Video quá ngắn, không đủ số lượng frame yêu cầu.")

            # Tính toán các chỉ số frame sẽ lấy mẫu đồng đều
            frame_indices = np.linspace(0, total_frames - 1, self.seq_len, dtype=int)
            
            face_sequence = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Chuyển đổi BGR (OpenCV) sang RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)

                # Cắt khuôn mặt
                face = self.mtcnn(img_pil)
                
                if face is not None:
                    # Chuyển từ PyTorch Tensor (3, 224, 224) về Numpy Array (224, 224, 3)
                    face_np = face.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    face_sequence.append(face_np)
                else:
                    # Bổ sung logic: Nếu frame này không thấy mặt, dùng mặt của frame trước đó (nếu có)
                    if len(face_sequence) > 0:
                        face_sequence.append(face_sequence[-1])

            cap.release()

            # Đảm bảo đủ số lượng frame, nếu thiếu thì padding bằng frame cuối
            while len(face_sequence) > 0 and len(face_sequence) < self.seq_len:
                face_sequence.append(face_sequence[-1])

            if len(face_sequence) == self.seq_len:
                return np.array(face_sequence)
            else:
                return None

        except Exception as e:
            print(f"[ERROR] Lỗi xử lý {video_path}: {str(e)}")
            return None

    def process_dataset(self, input_dir, output_dir):
        """
        Xử lý hàng loạt video trong thư mục và lưu dưới dạng .npy.
        """
        os.makedirs(output_dir, exist_ok=True)
        video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi'))]
        
        print(f"[INFO] Bắt đầu xử lý {len(video_files)} video...")
        
        success_count = 0
        for video_name in tqdm(video_files, desc="Đang trích xuất đặc trưng"):
            video_path = os.path.join(input_dir, video_name)
            output_path = os.path.join(output_dir, video_name.replace('.mp4', '.npy'))
            
            # Bỏ qua nếu đã xử lý
            if os.path.exists(output_path):
                continue
                
            sequence = self.process_video(video_path)
            
            if sequence is not None:
                np.save(output_path, sequence)
                success_count += 1
                
        print(f"[INFO] Hoàn thành. Số video xử lý thành công: {success_count}/{len(video_files)}")

# ----------------- Hướng dẫn sử dụng -----------------
# preprocessor = VideoPreprocessor(sequence_length=15, image_size=224, margin=20)
# preprocessor.process_dataset('FaceForensics/Original', 'Processed_Data/Original')
# preprocessor.process_dataset('FaceForensics/Face2Face', 'Processed_Data/Face2Face')
if __name__ == '__main__':
    # Khởi tạo bộ tiền xử lý
    preprocessor = VideoPreprocessor(sequence_length=15, image_size=224, margin=20)
    
    # Đường dẫn gốc tới thư mục dữ liệu trên ổ D của bạn
    base_data_dir = r"D:\FaceForensicsData"
    output_base_dir = r"D:\Project\Processed_Data"
    
    # -------------------------------------------------------------
    # BƯỚC 1: Tiền xử lý Video REAL (Thường nằm ở youtube/c23/videos)
    # -------------------------------------------------------------
    real_input_dir = os.path.join(base_data_dir, "original_sequences", "youtube", "c23", "videos")
    # Nếu thư mục gốc của bạn không có chữ 'youtube', hãy đổi thành:
    # real_input_dir = os.path.join(base_data_dir, "original_sequences", "c23", "videos")
    
    real_output_dir = os.path.join(output_base_dir, "Real")
    
    print("\n=================== XỬ LÝ VIDEO REAL ===================")
    if os.path.exists(real_input_dir):
        preprocessor.process_dataset(real_input_dir, real_output_dir)
    else:
        print(f"[CẢNH BÁO] Kiểm tra lại đường dẫn video Real: {real_input_dir}")
    
    # -------------------------------------------------------------
    # BƯỚC 2: Tự động chạy lặp qua cả 5 loại Video FAKE
    # -------------------------------------------------------------
    fake_methods = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
    
    print("\n=================== XỬ LÝ 5 LOẠI VIDEO FAKE ===================")
    for method in fake_methods:
        fake_input_dir = os.path.join(base_data_dir, "manipulated_sequences", method, "c23", "videos")
        fake_output_dir = os.path.join(output_base_dir, "Fake", method)
        
        if os.path.exists(fake_input_dir):
            print(f"\n[TIẾN TRÌNH] Đang xử lý tập: {method}")
            preprocessor.process_dataset(fake_input_dir, fake_output_dir)
        else:
            print(f"[CẢNH BÁO] Bỏ qua, không tìm thấy thư mục: {fake_input_dir}")