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

本システムは、ユーザーがアップロードした画像データを利用して画像分類AIモデルを学習し、学習済みモデルを .tflite 形式で出力する学習サービスです。フロントエンドは LiteRT.js から呼び出します。

FastAPI を利用して API を提供し、学習完了後に Go バックエンドへ通知を行います。

---

## 主な機能

* 複数クラス画像分類モデルの学習
* PyTorch (timm) による転移学習
* .tflite 形式への変換（変換前の元モデルも同梱）
* 学習済みモデルの ZIP 圧縮
* Go サーバーへの自動通知（コールバック）

---

## 技術スタック

### Web Framework

* FastAPI

### AI / Machine Learning

* PyTorch / timm（mobilenet_v3, efficientnet_lite4, mobilevit_v2 の3モデル共通基盤。`torch_models.py`）
* LiteRT（.tflite変換・TFLite Interpreterによる性能テスト評価）

TensorFlowは使用していない(`tensorflow[and-cuda]`とtorchのCUDAライブラリ要求が競合し
`pip install`が`ResolutionImpossible`になっていたため、2026-07-11に完全排除した)。

### Image Processing

* OpenCV
* NumPy

### Communication

* Requests

---

## 使用モデル

本システムでは以下の事前学習済みモデルを利用します。

| モデル              | timmモデル名                | 入力サイズ     | 配信用(変換済み) | アーカイブ用(変換前) |
| ---------------- | ------------------------ | --------- | ---------- | ----------- |
| mobilenet_v3     | `mobilenetv3_large_100`  | 224 × 224 | .tflite    | .pt         |
| efficientnet_lite4 | `tf_efficientnet_lite4` | 380 × 380 | .tflite    | .pt         |
| mobilevit_v2     | `mobilevitv2_100`        | 256 × 256 | .tflite    | .pt         |

すべてPyTorch/timm実装(`torch_models.py`)。分類ヘッドは生ロジットを返す(softmax層なし)ため、
評価・フロントエンド推論の双方で明示的にsoftmaxを適用する。

学習時には ImageNet の事前学習済み重みを利用した転移学習を行います。3モデルとも `.tflite` に変換してフロントエンド
(LiteRT.js)から呼び出す。Goバックエンドへ返すZIPには変換済み(.tflite)と変換前の元モデル(.pt)の両方を含める。

---

## .tflite / LiteRT.js への全面移行について(2026-07-10)

MobileViT-v2は元々TensorFlowの自作モデル(Conv+LayerNorm+MultiHeadAttentionを数層組み合わせただけの簡易実装で、
実在するMobileViT-v2アーキテクチャではなく、事前学習済み重みも実質読み込まれていなかった)だった。これを
PyTorch + `timm`(`mobilevitv2_100`, ImageNet事前学習済み)に置き換え、`litert-torch`(旧`ai-edge-torch`)で
`.tflite` に変換する構成にした。

その後、MobileNetV3Large / EfficientNetB4 を含む3モデルすべてを `.tflite` + LiteRT.js に統一した
(従来はTensorFlow.js形式)。性能テスト機能(`/test`)も3モデル共通のTFLite Interpreter評価に統一している。

- 旧TF版MobileViT-v2(ロールバック用にアーカイブ): `legacy/mobilevit_v2_tf/`

## TensorFlowの完全排除・PyTorch統一について(2026-07-11)

`requirements.txt` に `tensorflow[and-cuda]` と `torch` を同居させると、両者が要求する
`nvidia-cublas-cu12` 等CUDAライブラリの厳密なバージョン指定(`==`同士)が競合し、`pip install` が
`ResolutionImpossible` で失敗するようになった。これを構造的に解消するため、残っていた
`mobilenet_v3`(`MobileNetV3Large`) / `efficientnet_lite4`(誤って`EfficientNetB4`を使用していた)
のTensorFlow/Keras実装もPyTorch/timmへ移行し、`tensorflow` を依存関係から完全に排除した。
`main.py` の `/analyze` エンドポイント(ResNet50特徴抽出器)もPyTorch/timmに置き換えている。
外部インターフェース(FastAPIエンドポイント、Goへのコールバック形式、学習曲線のキー名)は変更していない。

- 新実装: `torch_models.py`(旧`mobilevit_v2_torch.py`。3モデル共通のPyTorch/timm学習・変換基盤に汎用化)
- `efficientnet_lite4` の入力サイズは380×380に変更(本物のEfficientNet-Lite4のネイティブ解像度。
  旧実装は別アーキテクチャを誤って使っていたため300×300だった)
- 旧TF版CNN実装(ロールバック用にアーカイブ): `legacy/tf_cnn_models/`
- 既知の注意点: `torch==2.6.0`は`pip install`自体は通るが、`litert-torch`が依存する`torchao`が
  `torch.utils._pytree.register_constant`(torch 2.6系には存在しない)を要求するため実行時に
  クラッシュする。`litert-torch`と実際に組み合わせて動作確認済みの`torch==2.10.0`を使うこと。

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

学習済みモデルは .tflite 形式(配信用)+ 変換前の元モデル(アーカイブ用)で保存されます。

出力先

```text
exported_models/
└── user_id/
    ├── mobilenet_v3/model.tflite       # フロント配信用
    ├── mobilenet_v3.tflite             # Go /test 評価用(model.tfliteと同一)
    ├── mobilenet_v3.pt                 # 変換前の元モデル(PyTorch state_dict, アーカイブ用)
    ├── efficientnet_lite4/model.tflite
    ├── efficientnet_lite4.tflite
    ├── efficientnet_lite4.pt
    ├── mobilevit_v2/model.tflite
    ├── mobilevit_v2.tflite
    ├── mobilevit_v2.pt
    └── label_map.json
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
├── models_config.py  # 1. AIモデルの構造定義(timm_name等)
├── torch_models.py   # 2. 3モデル共通のPyTorch/timm学習・前処理・.tflite変換基盤
├── ai_logic.py        # 3. 学習ループの呼び出し、テスト評価の重たいロジック
├── main.py            # 4. FastAPIの起動・APIルート・エンドポイント（軽量化）
└── .env               # 環境変数（既存のまま）
```

---

## 必要ライブラリ

```bash
pip install fastapi
pip install uvicorn
pip install torch torchvision timm
pip install litert-torch ai-edge-litert
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
* cuDNN対応版 PyTorch(`torch==2.10.0`)

混合精度学習（AMP）を有効化しており、VRAM使用量と学習速度を最適化しています(`torch_models.py`の
`setup_precision_torch()`がTensor Core対応GPUを検出して自動的に有効化)。

```python
with torch.cuda.amp.autocast(enabled=USE_AMP):
    ...
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



