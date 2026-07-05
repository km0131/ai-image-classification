MobileNetV3-Large（超軽量・モバイル向けCNNの代表）

EfficientNet-Lite4（精度と軽さのバランスが良いCNN）

MobileViT-v2 または TinyViT（ViTを軽量化したモデル）
にモデルを変更する。

# Pythonの仮想環境構築
```Bash
python3 -m venv .venv
```

# 仮想環境の有効化
```Bash
source .venv/bin/activate
```

# ライブラーのインストール
```Bash
# pipの更新
pip install --upgrade pip

# ライブラリの一括インストール
pip install -r requirements.txt
```

# AI Model Training Service

## 概要

本システムは、ユーザーがアップロードした画像データを利用して画像分類AIモデルを学習し、学習済みモデルを TensorFlow.js 形式で出力する学習サービスです。

FastAPI を利用して API を提供し、学習完了後に Go バックエンドへ通知を行います。

---

## 主な機能

* 複数クラス画像分類モデルの学習
* TensorFlow / Keras による転移学習
* TensorFlow.js 形式への変換
* 学習済みモデルの ZIP 圧縮
* Go サーバーへの自動通知（コールバック）

---

## 技術スタック

### Web Framework

* FastAPI

### AI / Machine Learning

* TensorFlow
* Keras
* TensorFlow.js

### Image Processing

* OpenCV
* NumPy

### Communication

* Requests

---

## 使用モデル

本システムでは以下の事前学習済みモデルを利用します。

| モデル              | 入力サイズ     |
| ---------------- | --------- |
| EfficientNetB4   | 300 × 300 |
| MobileNetV3Large | 224 × 224 |
| ConvNeXtTiny     | 256 × 256 |

学習時には ImageNet の事前学習済み重みを利用した転移学習を行います。

---

## API

### POST /process

画像データを受け取り、AIモデルを学習します。

#### リクエスト

multipart/form-data

##### パラメータ

| 名前             | 内容       |
| -------------- | -------- |
| user_id        | ユーザーID   |
| explanation_id | 説明ID     |
| label_0        | ラベル名     |
| images_0       | ラベル0の画像群 |
| label_1        | ラベル名     |
| images_1       | ラベル1の画像群 |
| ...            | ...      |

#### 送信例

```text
label_0 = cat
images_0 = cat1.jpg, cat2.jpg

label_1 = dog
images_1 = dog1.jpg, dog2.jpg
```

---

## 学習処理

### 1. 画像受信

アップロードされた画像を OpenCV により読み込みます。

### 2. 前処理

* RGB変換
* 正規化（0〜1）
* リサイズ

### 3. モデル学習

各モデルごとに学習を実施します。

```python
epochs=2
batch_size=32
optimizer=Adam(1e-5)
```

---

## モデル出力

学習済みモデルは TensorFlow.js 形式で保存されます。

出力先

```text
exported_models/
└── user_id/
    ├── effnet_lite4/
    ├── mobilenet_v3/
    └── convnext_tiny/
```

各フォルダには以下が生成されます。

```text
model.json
group1-shard1ofN.bin
```

---

## コールバック処理

学習完了後、生成されたモデルを ZIP 圧縮して Go サーバーへ送信します。

### 通知先

```text
/api/callback/model_ready
```

### 送信データ

| 名前             | 内容      |
| -------------- | ------- |
| user_id        | ユーザーID  |
| explanation_id | 説明ID    |
| model_zip      | ZIPファイル |

---

## ディレクトリ構成

```text
.
├── models_config.py  # 1. AIモデルの構造定義・GPU設定
├── ai_logic.py       # 2. 学習、TFJS変換、テスト評価の重たいロジック
├── main.py           # 3. FastAPIの起動・APIルート・エンドポイント（軽量化）
└── .env              # 環境変数（既存のまま）
```

---

## 必要ライブラリ

```bash
pip install fastapi
pip install uvicorn
pip install tensorflow
pip install tensorflowjs
pip install opencv-python
pip install numpy
pip install requests
```

または

```bash
pip install -r requirements.txt
```

---

## 起動方法

### 開発環境

```bash
python main.py
```

または

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## レスポンス例

### 成功

```json
{
  "status": "success",
  "user_id": "user001",
  "labels": {
    "0": "cat",
    "1": "dog"
  }
}
```

### 失敗

```json
{
  "error": "No images received"
}
```

---

## GPU利用について

本システムは NVIDIA GPU を利用した学習を想定しています。

推奨環境

* NVIDIA RTX 3060 Ti 以上
* CUDA 12系
* cuDNN対応版 TensorFlow

混合精度学習（Mixed Precision）を有効化しており、VRAM使用量と学習速度を最適化しています。

```python
mixed_precision.set_global_policy("mixed_float16")
```

---

---

- ローカル（docker-compose）起動例：リポジトリ直下で実行する。

```
docker compose up --build
podman-compose up --build
```

- コンテナの停止
```
podman-compose down
```


- コンテナの削除
```
sudo docker-compose down
```

---

## 注意事項

* クラスごとに十分な枚数の画像を用意してください。
* 学習データ数が少ない場合、精度が低下する可能性があります。
* 学習完了後に生成された ZIP ファイルは自動削除されます。
* コールバック先の Go サーバーが起動している必要があります。



