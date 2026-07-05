import os
import shutil
import zipfile
import json
import logging
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Header, BackgroundTasks  # ★BackgroundTasksをここに追加
from pydantic import BaseModel
from dotenv import load_dotenv

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
import logging
import requests
import threading

from models_config import MODEL_CONFIGS
import ai_logic

gpu_lock = threading.Lock()

load_dotenv()
PYTHON_API_SECRET = os.getenv("PYTHON_API_SECRET", "secure_python_analyze_secret_token_abc")

app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 特徴量抽出器の初期化
base_model = ResNet50(weights='imagenet', include_top=False)
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)


class AnalysisResponse(BaseModel):
    saturation: float
    brightness: float
    sharpness: float
    diversity_vector: list
    message: str


def verify_token(authorization: str):
    if authorization != f"Bearer {PYTHON_API_SECRET}":
        logger.warning(f"🔒 不正アクセスブロック: {authorization}")
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...), id: str = Form(...), authorization: str = Header(None)):
    verify_token(authorization)
    try:
        file_bytes = await file.read()
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None: raise HTTPException(status_code=400, detail="Invalid image")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = float(np.mean(hsv[:, :, 1])) / 255.0
        brightness = float(np.mean(hsv[:, :, 2])) / 255.0
        sharpness = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

        x = preprocess_input(np.expand_dims(cv2.resize(img, (224, 224)), axis=0))
        with gpu_lock:
            diversity_vector = feature_extractor.predict(x, verbose=0).flatten()[:2].tolist()

        return {"saturation": saturation, "brightness": brightness, "sharpness": sharpness,
                "diversity_vector": diversity_vector, "message": "Analysis successful"}
    except Exception as e:
        logger.error(f"🚨 エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", status_code=202)
async def process_ai(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        job_id: int = Form(...),
        authorization: str = Header(None)
):
    verify_token(authorization)
    user_id = str(job_id)
    temp_zip_path = f"/tmp/incoming_job_{user_id}.zip"

    with open(temp_zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(run_process_and_notify_go, temp_zip_path, job_id)

    return {"status": "accepted", "job_id": job_id}


def run_process_and_notify_go(temp_zip_path: str, job_id: int):
    user_id = str(job_id)
    extract_dir = f"/tmp/training_job_{user_id}"
    go_callback_url = os.getenv("GO_TRAIN_RESULT_CALLBACK_URL", "http://go-backend:8080/api/callback/model_ready")
    go_secret = os.getenv("CALLBACK_SECRET", "gcp_to_raspi_secure_callback_token_xyz")

    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        metadata_path = os.path.join(extract_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise RuntimeError("metadata.json not found")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        all_x, all_y, unique_labels = [], [], set()
        for entry in metadata:
            img_path = os.path.join(extract_dir, entry["filename"])
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                all_x.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                all_y.append(entry["label_id"])
                unique_labels.add(entry["label_id"])

        if not all_x:
            raise RuntimeError("No valid images processed")

        # ★GPUを使う学習処理をロックで直列化（run_training_and_callback内部で既にコールバック送信までしている点に注意）
        with gpu_lock:
            ai_logic.run_training_and_callback(extract_dir, user_id, all_x, all_y, unique_labels)

    except Exception as e:
        logger.error(f"[ERROR] 学習処理中にエラーが発生しました (job_id={job_id}): {e}")
        try:
            requests.post(
                go_callback_url,
                json={"status": "error", "job_id": job_id, "detail": str(e)},
                headers={"Authorization": f"Bearer {go_secret}"},
                timeout=60,
            )
        except Exception as notify_err:
            logger.error(f"[ERROR] Goへのエラー通知にも失敗しました: {notify_err}")

    finally:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)


# 新設：Goサーバーから画像ZIPと.kerasを受け取ってテストするAPI
from fastapi import BackgroundTasks

# 新設：Goサーバーから画像ZIPと.kerasを受け取ってテストするAPI
# 即座に202を返し、実際の推論処理はバックグラウンドで行う。
# 結果はGoの受け取り用Webhook（/api/callback/test_result）へ後で通知する。
@app.post("/test", status_code=202)
async def test_ai(
        background_tasks: BackgroundTasks,
        file_zip: UploadFile = File(...),  # Goから送られる単一ZIP (images/, models/*.keras, models/label_map.json, test_itinerary.json)
        status_id: int = Form(...),
        authorization: str = Header(None)
):
    verify_token(authorization)
    temp_zip_path = f"/tmp/test_job_{status_id}.zip"

    # リクエストが閉じられる前に、ファイルの中身をディスクへ保存しておく
    with open(temp_zip_path, "wb") as b:
        shutil.copyfileobj(file_zip.file, b)

    # 重い処理（展開・推論）はバックグラウンドに回して即応答する
    background_tasks.add_task(run_test_and_notify_go, temp_zip_path, status_id)

    return {"status": "accepted", "status_id": status_id}


def run_test_and_notify_go(temp_zip_path: str, status_id: int):
    """
    バックグラウンドで実際のテスト処理（ZIP展開・推論・Goへの結果通知）を行う。
    /testエンドポイントからのリクエストとは切り離されているため、
    ここでの例外はレスポンスに反映されない → 失敗時もGoへ通知する。
    """
    extract_dir = f"/tmp/test_extract_{status_id}"
    go_callback_url = os.getenv("GO_TEST_RESULT_CALLBACK_URL", "http://go-backend:8080/api/callback/test_result")
    go_secret = os.getenv("CALLBACK_SECRET", "gcp_to_raspi_secure_callback_token_xyz")

    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    try:
        # ZIPを展開
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Go側 model.TestItinerary に対応（"models": {model_name: [entry, ...]}）
        itinerary_path = os.path.join(extract_dir, "test_itinerary.json")
        if not os.path.exists(itinerary_path):
            raise RuntimeError("test_itinerary.json not found")
        with open(itinerary_path, "r", encoding="utf-8") as f:
            itinerary = json.load(f)

        models_meta = itinerary.get("models", {})
        if not models_meta:
            raise RuntimeError("No models found in test_itinerary.json")

        # ラベル対応表（idx <-> 実際のラベルID）を読み込む
        models_dir = os.path.join(extract_dir, "models")
        label_map_path = os.path.join(models_dir, "label_map.json")
        if not os.path.exists(label_map_path):
            raise RuntimeError("label_map.json not found")
        with open(label_map_path, "r", encoding="utf-8") as f:
            raw_label_map = json.load(f)
        idx_to_label = {int(k): v for k, v in raw_label_map.items()}
        label_to_idx = {v: k for k, v in idx_to_label.items()}

        summary = {}  # モデル名 -> {accuracy, loss, total_images}

        TEST_MAX_IMAGES = int(os.getenv("TEST_MAX_IMAGES", "0"))

        # モデルごとにループ（model_nameはここ、辞書のキーから取得）
        for model_name, entries in models_meta.items():
            keras_path = os.path.join(models_dir, f"{model_name}.keras")
            if not os.path.exists(keras_path):
                logger.warning(f"model file not found for {model_name}, skipping")
                continue

            target_entries = entries[:TEST_MAX_IMAGES] if TEST_MAX_IMAGES > 0 else entries

            test_x, true_indices = [], []
            valid_entries = []
            for entry in target_entries :
                img_path = os.path.join(extract_dir, entry["filename"])
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                true_label_id = entry["true_label_id"]
                if true_label_id not in label_to_idx:
                    logger.warning(f"unknown true_label_id {true_label_id} for model {model_name}, skip")
                    continue
                test_x.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                true_indices.append(label_to_idx[true_label_id])
                valid_entries.append(entry)

            if not test_x:
                logger.warning(f"no valid test images for model {model_name}, skipping")
                continue

            with gpu_lock:
                result = ai_logic.evaluate_test_model(keras_path, extract_dir, model_name, test_x, true_indices)

            # 予測インデックス -> 実際のラベルIDに変換してitineraryに書き戻す
            for entry, pred_idx, conf in zip(valid_entries, result["predictions"], result["confidences"]):
                entry["predicted_label_id"] = idx_to_label[pred_idx]
                entry["confidence"] = float(conf)

            summary[model_name] = {
                "accuracy": result["accuracy"],
                "loss": result["loss"],
                "total_images": len(test_x),
            }

        result_payload = {
            "status": "success",
            "status_id": status_id,
            "summary": summary,      # モデル別集計 -> StudentTestJobModel用
            "itinerary": itinerary,  # 画像単位の予測結果込み -> StudentTestResultSnapshot用
        }

        # ★Goの受け取り用Webhookへ結果を通知
        requests.post(
            go_callback_url,
            json=result_payload,
            headers={"Authorization": f"Bearer {go_secret}"},
            timeout=60,
        )
        logger.info(f"[INFO] テスト結果をGoへ通知しました (status_id={status_id})")

    except Exception as e:
        logger.error(f"[ERROR] テスト実行中にエラーが発生しました (status_id={status_id}): {e}")
        # 失敗時もGoへ通知しておくことで、ステータスを"failed"などに更新できる
        try:
            requests.post(
                go_callback_url,
                json={"status": "error", "status_id": status_id, "detail": str(e)},
                headers={"Authorization": f"Bearer {go_secret}"},
                timeout=60,
            )
        except Exception as notify_err:
            logger.error(f"[ERROR] Goへのエラー通知にも失敗しました: {notify_err}")

    finally:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)