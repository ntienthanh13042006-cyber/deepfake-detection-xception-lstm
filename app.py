import streamlit as st
import time
import random

# Cấu hình giao diện trang web
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🛡️",
    layout="wide"
)

# --- PHẦN GIAO DIỆN (FRONTEND) ---
st.title("🛡️ Hệ Thống Phát Hiện Video Giả Mạo (Deepfake Detection)")
st.markdown("### Đồ án tốt nghiệp - Nhóm nghiên cứu AI & Mạng máy tính")
st.markdown("---")

# Thanh sidebar bên trái (Dấu ấn của nhóm)
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
    st.subheader("Thông tin hệ thống")
    st.info("Trạng thái: **Sẵn sàng Demo**")
    st.markdown("**Công nghệ sử dụng:**")
    st.markdown("- Spatial: Xception Net")
    st.markdown("- Temporal: Bi-LSTM")
    st.markdown("- Framework: PyTorch & Streamlit")
    st.markdown("---")
    st.markdown("*Lưu ý: Đây là bản demo giao diện luồng xử lý hệ thống.*")

# Khu vực chính: Tải video lên
uploaded_file = st.file_uploader("Tải lên một đoạn video cần kiểm tra (.mp4, .avi)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎥 Video Gốc")
        st.video(uploaded_file)
        
    with col2:
        st.subheader("⚙️ Kết Quả Phân Tích")
        
        # Nút bấm chạy phân tích
        if st.button("Bắt đầu Phân tích Video", type="primary"):
            
            # Hiển thị thanh tiến trình giả lập quá trình xử lý
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Bước 1/3: Đang đọc và trích xuất khung hình...")
            for i in range(33):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
                
            status_text.text("Bước 2/3: Đang nhận diện và cắt khuôn mặt (Face Cropping)...")
            for i in range(33, 66):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
                
            status_text.text("Bước 3/3: Mô hình Xception + Bi-LSTM đang phân tích chuỗi...")
            for i in range(66, 100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
                
            status_text.empty()
            progress_bar.empty()
            
            # Giả lập kết quả trả về (Random Real hoặc Fake để demo)
            # Khi bảo vệ chính thức, chỗ này sẽ gọi model thật dự đoán
            is_fake = random.choice([True, False])
            confidence = random.uniform(94.5, 99.8)
            
            st.success("✅ Phân tích hoàn tất!")
            
            # Hiển thị kết quả nổi bật
            if is_fake:
                st.error(f"🚨 CẢNH BÁO: PHÁT HIỆN VIDEO GIẢ MẠO (DEEPFAKE)!")
                st.metric(label="Mức độ tin cậy", value=f"{confidence:.2f}%", delta="Độ rủi ro cao")
            else:
                st.success(f"✅ KẾT QUẢ: VIDEO CHÂN THỰC (REAL)")
                st.metric(label="Mức độ tin cậy", value=f"{confidence:.2f}%", delta="An toàn")
                
            # Biểu đồ mô phỏng phân phối lỗi qua các khung hình
            st.markdown("#### 📊 Biểu đồ dao động bất thường qua 15 khung hình:")
            chart_data = [random.uniform(0.1, 0.9) for _ in range(15)]
            st.line_chart(chart_data)