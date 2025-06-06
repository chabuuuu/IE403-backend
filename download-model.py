import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Tải biến môi trường từ file .env (nếu có)
load_dotenv()

# --- CẤU HÌNH ---

# 1. ID của model trên Hugging Face Hub
model_id = "Viet-Mistral/Vistral-7B-Chat"

# 2. Đường dẫn đến thư mục bạn muốn LƯU model vào
#    - Đối với Google Colab, hãy dùng đường dẫn trên Drive để lưu vĩnh viễn
#    - Đối với máy tính cá nhân, hãy chọn một thư mục có đủ dung lượng
# LƯU Ý: Model này rất lớn (~15 GB), hãy đảm bảo bạn có đủ dung lượng!
save_directory = "./model/Vistral-7B-Chat"

# 3. Lấy token từ biến môi trường
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("CẢNH BÁO: Không tìm thấy Hugging Face Token. Có thể không tải được các model yêu cầu xác thực.")

# --- THỰC HIỆN TẢI ---

print(f"Đang chuẩn bị tải model: {model_id}")
print(f"Sẽ lưu vào thư mục: {save_directory}")

# Tạo thư mục nếu chưa tồn tại
os.makedirs(save_directory, exist_ok=True)

try:
    # snapshot_download sẽ tải toàn bộ "snapshot" của repo model
    snapshot_download(
        repo_id=model_id,
        local_dir=save_directory,
        local_dir_use_symlinks=False, # Đặt là False để sao chép file trực tiếp
        token=hf_token,
        # resume_download=True # Bật lại nếu quá trình tải bị ngắt quãng
    )
    print("\n🎉 Tải model thành công! 🎉")
    print(f"Model đã được lưu tại: {save_directory}")

except Exception as e:
    print(f"\nĐã xảy ra lỗi trong quá trình tải: {e}")