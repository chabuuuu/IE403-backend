# Sử dụng base image Python chính thức
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các gói phụ thuộc của hệ thống
# Cần thiết cho việc biên dịch một số thư viện Python
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Sao chép file requirements trước để tận dụng Docker cache
COPY requirements.txt .

# Cài đặt các thư viện Python
# Thêm --default-timeout để tránh lỗi khi tải các gói lớn
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Sao chép toàn bộ mã nguồn của ứng dụng vào image
# Bao gồm main.py, download-model.py, .env, ...
COPY . /app

# --- THAY ĐỔI Ở ĐÂY ---
# Chạy script để tải model. Model sẽ được lưu vào trong layer của image.
# Bước này sẽ mất rất nhiều thời gian khi build image lần đầu.
RUN python download-model.py

# Mở cổng mặc định của FastAPI
EXPOSE 8000

# Lệnh để chạy ứng dụng FastAPI khi container khởi động
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]