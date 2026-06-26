# 軽量なPythonイメージを使用
FROM python:3.11-slim

# 作業ディレクトリの設定
WORKDIR /app

# 画像処理に必要なライブラリをインストール（OpenCVに必要な依存関係）
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 依存ライブラリのインストール
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -v -r requirements.txt

# ソースコードをコピー
COPY . .

# FastAPIをポート5000で起動
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]