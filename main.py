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

import timm
import torch
import torch_models as tm
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

# 特徴量抽出器の初期化(DINOv2。「似た画像は近く、違う画像は遠くに配置される」ことを自己教師あり学習で
# 直接学習しているため、ImageNet分類用のResNet50よりdiversity_vector用途に適している)
# num_classes=0で分類ヘッドを除去し埋め込みベクトルをそのまま取得。img_size=224でデフォルトの
# 518x518より計算コストを抑える。
feature_extractor = timm.create_model(
    "vit_small_patch14_dinov2.lvd142m",
    pretrained=True,
    num_classes=0,
    img_size=224,
)
feature_extractor.eval().to(tm.DEVICE)

# 既存の共通関数をそのまま利用するが、resolve_data_config()はimg_size=224の指定を無視して
# チェックポイントのネイティブ解像度(518x518)を返してしまうため、height/widthだけ実際の
# 推論サイズに上書きする(mean/stdはこの調整の影響を受けないため既存値のまま使う)。
feature_preprocess_cfg = tm.get_preprocess_config(feature_extractor)
feature_preprocess_cfg["height"] = 224
feature_preprocess_cfg["width"] = 224


class AnalysisResponse(BaseModel):
    saturation: float
    brightness: float
    sharpness: float
    diversity_vector: list
    message: str


class ReduceDiversityRequest(BaseModel):
    vectors: list[list[float]]


class ReduceDiversityResponse(BaseModel):
    points: list[list[float]]


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

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (feature_preprocess_cfg["width"], feature_preprocess_cfg["height"]))
        x = tm.prepare_inputs(np.expand_dims(resized, axis=0).astype(np.float32), feature_preprocess_cfg)
        with gpu_lock, torch.no_grad():
            # 384次元の埋め込みをそのまま返す(先頭2要素への機械的な切り出しは意味のある2次元化ではないため廃止)。
            # 2次元への圧縮(PCA)はジョブ内の全画像が揃った時点で /reduce_diversity がまとめて行う。
            diversity_vector = feature_extractor(x.to(tm.DEVICE)).cpu().numpy().flatten().tolist()

        return {"saturation": saturation, "brightness": brightness, "sharpness": sharpness,
                "diversity_vector": diversity_vector, "message": "Analysis successful"}
    except Exception as e:
        logger.error(f"🚨 エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def _pca_2d(vectors: np.ndarray) -> np.ndarray:
    """SVDベースのPCAで高次元ベクトル群を2次元に射影する(scikit-learn非依存)。
    データ点が2点未満、または分散が潰れている場合は原点(0,0)を返す。"""
    n = vectors.shape[0]
    if n < 2:
        return np.zeros((n, 2), dtype=np.float64)
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((n, 2), dtype=np.float64)
    k = min(2, vt.shape[0])
    projected = centered @ vt[:k].T
    if k < 2:
        projected = np.pad(projected, ((0, 0), (0, 2 - k)))
    return projected


# ジョブに属する全画像のdiversity_vector(高次元埋め込み)をまとめてPCAで2次元化する。
# Go側 GetImageEvaluationDB が表示リクエストのたびに呼び出す(バッチ処理・永続化はしない)。
@app.post("/reduce_diversity", response_model=ReduceDiversityResponse)
async def reduce_diversity(payload: ReduceDiversityRequest, authorization: str = Header(None)):
    verify_token(authorization)
    try:
        if not payload.vectors:
            return {"points": []}

        lengths = {len(v) for v in payload.vectors}
        if len(lengths) > 1:
            raise HTTPException(status_code=400, detail=f"vectors have inconsistent lengths: {sorted(lengths)}")

        arr = np.array(payload.vectors, dtype=np.float64)
        points = _pca_2d(arr)
        return {"points": points.tolist()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🚨 reduce_diversity エラー: {str(e)}")
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
            resp = requests.post(
                go_callback_url,
                json={"status": "error", "job_id": job_id, "detail": str(e)},
                headers={"Authorization": f"Bearer {go_secret}"},
                timeout=60,
            )
            resp.raise_for_status()
            logger.info(f"[INFO] Goへのエラー通知が完了しました (job_id={job_id}, status_code={resp.status_code})")
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
        # 3モデルとも.tfliteへ移行済み（Go側 test_worker.go も.tfliteのみをZIPに同梱する）
        for model_name, entries in models_meta.items():
            tflite_path = os.path.join(models_dir, f"{model_name}.tflite")
            if not os.path.exists(tflite_path):
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
                result = ai_logic.evaluate_test_model(tflite_path, extract_dir, model_name, test_x, true_indices)

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
            "detail": "",            # Go側 TestResultCallbackInput.Detail と揃える(成功時は空文字)
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