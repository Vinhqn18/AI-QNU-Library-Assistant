FROM python:3.10-slim

WORKDIR /code

# Cập nhật pip và cài đặt các gói hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY ./requirements.txt /code/requirements.txt

# --no-cache-dir đã có, thêm --default-timeout=100 nếu mạng chậm
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
