FROM python:3.10-slim-bookworm
WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    autoconf \
    libpq-dev \
    libmagic1 \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*



# Create and activate virtual environment
# RUN python3 -m venv venv


RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


RUN pip install --no-cache-dir --verbose \
    langchain \
    streamlit \
    openai \
    langchain-community \
    "unstructured[all-docs]"

# Install requirements from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pytesseract paddleocr google-cloud-vision

EXPOSE 8000
# CMD ["/app/venv/bin/streamlit", "run", "app.py", "--server.port", "8000"]
CMD ["streamlit", "run", "login.py", "--server.port", "8000"]
