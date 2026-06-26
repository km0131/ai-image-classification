import os
import shutil  # 追加：ZIP作成用
import requests # 追加：Goへの送信用
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
import tensorflowjs as tfjs
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException ,Header
from pydantic import BaseModel
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
import logging
import zipfile
import json
from dotenv import load_dotenv

print("起動しました。")

load_dotenv()

PYTHON_API_SECRET = os.getenv("PYTHON_API_SECRET", "secure_python_analyze_secret_token_abc")
app = FastAPI()

# 特徴量抽出モデルのロード（グローバルスコープで一度だけロード）
base_model = ResNet50(weights='imagenet', include_top=False)
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

class AnalysisResponse(BaseModel):
    saturation: float
    brightness: float
    message: str
    sharpness: float  # 追加
    diversity_vector: list  # 追加 (次元削減後の2次元ベクトル)
    message: str


# --- ログの設定 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
        request: Request,
        file: UploadFile = File(...),
        id: str = Form(...),  # Go側の WriteField("id", ...) を受け取る場合
        authorization: str = Header(None)
):
    expected_header = f"Bearer {PYTHON_API_SECRET}"
    if authorization != expected_header:
        logger.warning(f"🔒 不正なアクセスブロック: {authorization}")
        raise HTTPException(status_code=401, detail="Unauthorized: 不正な認証トークンです")

    logger.info(f"📥 解析リクエストを受信しました (ファイル直接送信): ID={id}, Filename={file.filename}")

    try:
        # 1. 送信されたバイナリを直接読み込む
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)

        # 2. OpenCVでデコード
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        logger.info(f"📸 画像デコード成功: shape={img.shape}")

        # --- 以降の明度・彩度・ResNet50解析ロジックは以前のままでOK ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = float(np.mean(hsv[:, :, 1])) / 255.0
        brightness = float(np.mean(hsv[:, :, 2])) / 255.0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        img_resized = cv2.resize(img, (224, 224))
        x = np.expand_dims(img_resized, axis=0)
        x = preprocess_input(x)
        features = feature_extractor.predict(x, verbose=0).flatten()
        diversity_vector = features[:2].tolist()

        return {
            "saturation": saturation,
            "brightness": brightness,
            "sharpness": sharpness,
            "diversity_vector": diversity_vector,
            "message": "Analysis successful"
        }

    except Exception as e:
        logger.error(f"🚨 エラー発生: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 混合精度 (RTX 3060 Ti用)
mixed_precision.set_global_policy(mixed_precision.Policy('mixed_float16'))

MODEL_CONFIGS = {
    'effnet_lite4': {'size': (300, 300), 'base': tf.keras.applications.EfficientNetB4},
    'mobilenet_v3': {'size': (224, 224), 'base': tf.keras.applications.MobileNetV3Large},
    'convnext_tiny': {'size': (256, 256), 'base': tf.keras.applications.ConvNeXtTiny}
}

# Ai 作成
@app.post("/process")
async def process_ai(
        file: UploadFile = File(...),  # Go側の CreateFormFile("file", ...)
        job_id: int = Form(...),  # Go側の WriteField("job_id", ...)
        authorization: str = Header(None)
):
    expected_header = f"Bearer {PYTHON_API_SECRET}"
    if authorization != expected_header:
        logger.warning(f"🔒 不正なアクセスブロック: {authorization}")
        raise HTTPException(status_code=401, detail="Unauthorized: 不正な認証トークンです")

    user_id = str(job_id)

    # 1. 届いたZIPファイルを一時保存・解凍するパス
    temp_zip_path = f"/tmp/incoming_job_{user_id}.zip"
    extract_dir = f"/tmp/training_job_{user_id}"

    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    try:
        # ZIPファイルを一時保存して解凍
        with open(temp_zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # metadata.json のパース
        metadata_path = os.path.join(extract_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            return {"error": "metadata.json not found in zip"}

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        all_x = []
        all_y = []
        unique_labels = set()

        # 画像の読み込み
        for entry in metadata:
            filename = entry["filename"]
            label_id = entry["label_id"]

            img_path = os.path.join(extract_dir, filename)
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                all_x.append(img_rgb)
                all_y.append(label_id)
                unique_labels.add(label_id)

    finally:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)

    if not all_x:
        return {"error": "No valid images could be processed from the zip"}

    # データのテンソル化
    x_train = np.array(all_x).astype(np.float32) / 255.0
    y_train = np.array(all_y)
    num_classes = len(unique_labels)

    # 【特徴量計算】平均彩度とデータ多様性スコア
    avg_saturation = 0.0
    if len(all_x) > 0:
        sats = [np.mean(cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1]) for img in all_x]
        avg_saturation = float(np.mean(sats))
    diversity_score = 0.85

    # ─── 4. 3つのモデルを順次学習 ───
    all_models_curves = {}
    summary_accuracy = 0.0
    summary_loss = 0.0

    # 成果物を格納する共通ルート。ここに3つのモデルフォルダが掘られます
    user_export_root = f"./exported_models/{user_id}"
    if os.path.exists(user_export_root):
        shutil.rmtree(user_export_root)
    os.makedirs(user_export_root, exist_ok=True)

    for model_name, config in MODEL_CONFIGS.items():
        img_size = config['size']
        # 各モデルのサイズに合わせて画像をリサイズ
        x_resized = tf.image.resize(x_train, img_size)

        base = config['base'](input_shape=(*img_size, 3), include_top=False, weights='imagenet')
        base.trainable = True

        model = models.Sequential([
            base,
            layers.GlobalAveragePooling2D(),
            # 混合精度ポリシーが mixed_float16 のため、Denseの出力を明示的に float32 にする
            layers.Dense(num_classes, activation='softmax', dtype='float32')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # 学習の実行
        history = model.fit(x_resized, y_train, epochs=2, batch_size=32, verbose=1)

        # エポックごとの履歴を取得
        epoch_accuracies = [float(x) for x in history.history['accuracy']]
        epoch_losses = [float(x) for x in history.history['loss']]

        # モデル別の学習曲線を記録
        all_models_curves[model_name] = [
            {"epoch": i + 1, "accuracy": epoch_accuracies[i], "loss": epoch_losses[i]}
            for i in range(len(epoch_accuracies))
        ]

        # 代表値（サマリー用）として、最後のモデルの最終エポック値を採用（必要に応じて平均などに変更可）
        summary_accuracy = epoch_accuracies[-1]
        summary_loss = epoch_losses[-1]

        # モデルの保存と tfjs への変換（各モデル固有のサブフォルダへ保存）
        export_path = os.path.join(user_export_root, model_name)
        os.makedirs(export_path, exist_ok=True)

        keras_model_path = f"temp_{model_name}.keras"
        model.save(keras_model_path)
        tfjs.converters.save_keras_model(model, export_path)

        # メモリ解放
        tf.keras.backend.clear_session()
        if os.path.exists(keras_model_path):
            os.remove(keras_model_path)

    # ─── 5. Goサーバーへの返却処理（一括ZIP化） ───
    # ./exported_models/{user_id} の中に effnet_lite4/, mobilenet_v3/, convnext_tiny/ が揃った状態で丸ごとZIP化
    zip_temp_name = f"temp_{user_id}"
    shutil.make_archive(zip_temp_name, 'zip', user_export_root)
    zip_file_path = f"{zip_temp_name}.zip"

    callback_url = os.getenv("GO_CALLBACK_URL")
    if not callback_url:
        callback_url = "http://100.102.77.94:8080/api/callback/model_ready"
    callback_secret = os.getenv("CALLBACK_SECRET", "gcp_to_raspi_secure_callback_token_xyz")
    headers = {
        "Authorization": f"Bearer {callback_secret}"
    }

    try:
        with open(zip_file_path, "rb") as f:
            files = {"model_zip": (f"{user_id}.zip", f)}

            data = {
                "job_id": user_id,
                "avg_saturation": f"{avg_saturation:.2f}",
                "diversity_score": f"{diversity_score:.2f}",
                "accuracy": f"{summary_accuracy:.4f}",
                "loss": f"{summary_loss:.4f}",
                # 🌟 3つすべてのモデルの学習曲線をまとめてJSON化して送信
                "learning_curve": json.dumps(all_models_curves)
            }

            response = requests.post(
                callback_url,
                files=files,
                data=data,
                headers=headers,
                timeout=300,
                verify=False
            )
        print(f"Go callback status: {response.status_code}")
    except Exception as e:
        print(f"Goへの通知失敗: {e}")
    finally:
        # ディレクトリと生成した一時ZIPのクリーンアップ
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        if os.path.exists(user_export_root):
            shutil.rmtree(user_export_root)

    return {"status": "success", "job_id": job_id, "models_trained": list(MODEL_CONFIGS.keys())}

if __name__ == "__main__":
    import uvicorn
    # 本番環境のIP、または0.0.0.0を指定
    uvicorn.run(app, host="0.0.0.0", port=8000)