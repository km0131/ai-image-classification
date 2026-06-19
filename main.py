from fastapi import FastAPI, File, Form, UploadFile, Request
import numpy as np
import cv2
import os
import shutil  # 追加：ZIP作成用
import requests # 追加：Goへの送信用
from typing import List, Dict
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
import tensorflowjs as tfjs

# 混合精度 (RTX 3060 Ti用)
mixed_precision.set_global_policy(mixed_precision.Policy('mixed_float16'))

app = FastAPI()

MODEL_CONFIGS = {
    'effnet_lite4': {'size': (300, 300), 'base': tf.keras.applications.EfficientNetB4},
    'mobilenet_v3': {'size': (224, 224), 'base': tf.keras.applications.MobileNetV3Large},
    'convnext_tiny': {'size': (256, 256), 'base': tf.keras.applications.ConvNeXtTiny}
}

@app.post("/process")
async def process_ai(request: Request):
    form = await request.form()
    explanation_id = form.get("explanation_id")
    user_id = form.get("user_id", "default_user")
    all_x = []
    all_y = []
    label_map = {}

    # ラベルと画像をパッキング
    idx = 0
    while f"label_{idx}" in form:
        label_name = form.get(f"label_{idx}")
        label_map[idx] = label_name
        
        upload_files = form.getlist(f"images_{idx}")
        for f in upload_files:
            contents = await f.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                all_x.append(img_rgb)
                all_y.append(idx)
        idx += 1

    if not all_x:
        return {"error": "No images received"}

    x_train = np.array(all_x).astype(np.float32) / 255.0
    y_train = np.array(all_y)
    num_classes = idx

    # 順次学習
    results = []
    for model_name, config in MODEL_CONFIGS.items():
        img_size = config['size']
        x_resized = tf.image.resize(x_train, img_size)

        base = config['base'](input_shape=(*img_size, 3), include_top=False, weights='imagenet')
        base.trainable = True
        model = models.Sequential([
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(num_classes, activation='softmax', dtype='float32')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x_resized, y_train, epochs=2, batch_size=32, verbose=1)

        export_path = f"./exported_models/{user_id}/{model_name}"
        os.makedirs(export_path, exist_ok=True)
        keras_model_path = f"temp_{model_name}.keras"
        model.save(keras_model_path)
        tfjs.converters.save_keras_model(model, export_path)
        results.append(model_name)
        tf.keras.backend.clear_session()
        if os.path.exists(keras_model_path):
            os.remove(keras_model_path)

    # --- Goサーバーへの返却処理 ---
    user_export_root = f"./exported_models/{user_id}"
    zip_temp_name = f"temp_{user_id}"
    
    # ZIP化
    shutil.make_archive(zip_temp_name, 'zip', user_export_root)
    zip_file_path = f"{zip_temp_name}.zip"

    callback_url = f"http://100.102.77.94:8080/api/callback/model_ready"
    try:
        with open(zip_file_path, "rb") as f:
            files = {"model_zip": (f"{user_id}.zip", f)}
            data = {
                "user_id": user_id,
                "explanation_id": explanation_id # これを追加
            }
            # verify=False を試す（トンネル内でのプロトコル不一致回避）
            # タイムアウトを長めに設定
            response = requests.post(
                callback_url, 
            files=files, 
            data=data, 
            timeout=300, 
            verify=False 
        )
        print(f"Go callback status: {response.status_code}")
    except Exception as e:
        print(f"Goへの通知失敗: {e}")
    finally:
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)

    return {"status": "success", "user_id": user_id, "labels": label_map}

if __name__ == "__main__":
    import uvicorn
    # 本番環境のIP、または0.0.0.0を指定
    uvicorn.run(app, host="0.0.0.0", port=8000)