import os
import gc
import json
import shutil
import zipfile
import cv2
import numpy as np
import requests
import subprocess
# TensorFlowをimportする前に設定する必要がある
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

if os.getenv("FORCE_CPU_FOR_TEST", "false").lower() == "true":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras import layers, models
from models_config import MODEL_CONFIGS, EPOCH1, EPOCH2, BATCH1, BATCH2, VALIDATION, FINE_TUNE

def run_training_and_callback(extract_dir, user_id, all_x, all_y, unique_labels):
    # 1. データのテンソル化
    label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    y_train = np.array([label_map[label] for label in all_y], dtype=np.int32)
    num_classes = len(unique_labels)

    # 特徴量計算（彩度）
    avg_saturation = 0.0
    if len(all_x) > 0:
        sats = [np.mean(cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1]) for img in all_x]
        avg_saturation = float(np.mean(sats))
    diversity_score = 0.85

    all_models_curves = {}
    model_summary = {}
    summary_loss = 0.0
    user_export_root = f"./exported_models/{user_id}"

    if os.path.exists(user_export_root):
        shutil.rmtree(user_export_root)
    os.makedirs(user_export_root, exist_ok=True)

    # ★追加：3モデル共通のラベル対応表（idx → 元のラベルID）を書き出す
    #   テスト実行時にPython側で「予測インデックス→実際のラベルID」を逆引きするために必要
    label_order = sorted(unique_labels)
    label_map_export = {idx: label for idx, label in enumerate(label_order)}
    with open(os.path.join(user_export_root, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map_export, f, ensure_ascii=False, indent=2)

    # 3モデルループ
    for model_name, config in MODEL_CONFIGS.items():
        print(f"=== Training: {model_name} ===")
        img_size = config["size"]
        x_resized = np.array([cv2.resize(img, (img_size[1], img_size[0])) for img in all_x], dtype=np.float32)

        if model_name == "mobilenet_v3":
            x_resized = tf.keras.applications.mobilenet_v3.preprocess_input(x_resized)
        elif model_name == "efficientnet_lite4":
            x_resized = tf.keras.applications.efficientnet.preprocess_input(x_resized)
        elif model_name == "mobilevit_v2":
            x_resized = x_resized / 127.5 - 1.0

        base = config["base"](input_shape=(*img_size, 3), include_top=False, weights=config.get("weights"))
        base.trainable = False

        model = models.Sequential([
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(num_classes, activation="softmax", dtype="float32")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Phase 1
        history1 = model.fit(
            x_resized, y_train, epochs=EPOCH1, batch_size=BATCH1, validation_split=VALIDATION,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7, verbose=1)
            ], verbose=1
        )

        # Phase 2 Fine tuning
        if FINE_TUNE:
            base.trainable = True
            fine_tune_at = int(len(base.layers) * 0.7)
            for layer in base.layers[:fine_tune_at]:
                layer.trainable = False
            for layer in base.layers:
                if isinstance(layer, (layers.BatchNormalization, layers.LayerNormalization)):
                    layer.trainable = False

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            history2 = model.fit(
                x_resized, y_train, epochs=EPOCH2, batch_size=BATCH2, validation_split=VALIDATION,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True,
                                                     verbose=1),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-8,
                                                         verbose=1)
                ], verbose=1
            )

        # 履歴の統合
        history_dict = {k: history1.history[k] + history2.history[k] if FINE_TUNE else history1.history[k] for k in
                        history1.history.keys()}

        epoch_accuracies = [float(x) for x in history_dict["accuracy"]]
        epoch_losses = [float(x) for x in history_dict["loss"]]
        val_accuracies = [float(x) for x in history_dict["val_accuracy"]]
        val_losses = [float(x) for x in history_dict["val_loss"]]

        all_models_curves[model_name] = [
            {"epoch": i + 1, "accuracy": epoch_accuracies[i], "loss": epoch_losses[i],
             "val_accuracy": val_accuracies[i], "val_loss": val_losses[i]}
            for i in range(len(epoch_accuracies))
        ]

        best_epoch = np.argmin(val_losses)
        summary_loss = val_losses[best_epoch]
        model_summary[model_name] = {"accuracy": val_accuracies[best_epoch], "loss": val_losses[best_epoch]}

        # 🌟 TF.js へのコンバート処理
        export_path = os.path.join(user_export_root, model_name)
        os.makedirs(export_path, exist_ok=True)
        temp_h5_path = f"/tmp/temp_model_{user_id}_{model_name}.h5"
        model.save(temp_h5_path, include_optimizer=False, save_format="h5")
        native_keras_path = os.path.join(user_export_root, f"{model_name}.keras")
        model.save(native_keras_path, save_format="keras")

        try:
            cmd = ["tensorflowjs_converter", "--input_format", "keras", temp_h5_path, export_path]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"[{model_name}] TFJSへの変換成功")
        except subprocess.CalledProcessError as e:
            print(f"💥 TFJS変換失敗: {e.stderr}")
            raise e
        finally:
            if os.path.exists(temp_h5_path):
                os.remove(temp_h5_path)

        tf.keras.backend.clear_session()
        del model, base
        gc.collect()

    # Goサーバーへ一括ZIP送信
    zip_temp_name = f"temp_{user_id}"
    shutil.make_archive(zip_temp_name, 'zip', user_export_root)
    zip_file_path = f"{zip_temp_name}.zip"

    callback_url = os.getenv("GO_CALLBACK_URL", "http://100.102.77.94:8080/api/callback/model_ready")
    callback_secret = os.getenv("CALLBACK_SECRET", "gcp_to_raspi_secure_callback_token_xyz")

    try:
        with open(zip_file_path, "rb") as f:
            files = {"model_zip": (f"{user_id}.zip", f)}
            data = {
                "job_id": user_id,
                "avg_saturation": f"{avg_saturation:.2f}",
                "diversity_score": f"{diversity_score:.2f}",
                "accuracy": json.dumps(model_summary),
                "loss": f"{summary_loss:.4f}",
                "learning_curve": json.dumps(all_models_curves)
            }
            print("\n【デバッグ】Goバックエンドへデータを送信します:")
            print(f"送信先URL: {callback_url}")
            print(f"添付ファイル名: {user_id}.zip")
            print("--- 送信フォームデータ (JSON) ---")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            print("======================================================== \n")
            requests.post(callback_url, files=files, data=data, headers={"Authorization": f"Bearer {callback_secret}"},
                          timeout=300, verify=False)
    finally:
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
        if os.path.exists(user_export_root):
            shutil.rmtree(user_export_root)


# Goから届いた .keras モデルと画像ZIPを使って性能テストをするロジック
def evaluate_test_model(model_path, extract_dir, model_name, all_test_x, all_test_y):
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        return {"error": f"Unknown model_name: {model_name}"}

    img_size = config["size"]
    x_resized = np.array([cv2.resize(img, (img_size[1], img_size[0])) for img in all_test_x], dtype=np.float32)

    if model_name == "mobilenet_v3":
        x_resized = tf.keras.applications.mobilenet_v3.preprocess_input(x_resized)
    elif model_name == "efficientnet_lite4":
        x_resized = tf.keras.applications.efficientnet.preprocess_input(x_resized)
    elif model_name == "mobilevit_v2":
        x_resized = x_resized / 127.5 - 1.0

    y_test = np.array(all_test_y, dtype=np.int32)

    model = tf.keras.models.load_model(model_path)
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    test_batch_size = int(os.getenv("TEST_BATCH_SIZE", "8"))

    print(f"[{model_name}] evaluate開始: 画像枚数={len(all_test_x)}, batch_size={test_batch_size}")

    # 集計値（従来通り）
    loss, accuracy = model.evaluate(x_resized, y_test,batch_size=test_batch_size, verbose=0)

    # 追加：画像ごとの予測結果（StudentTestResultSnapshot用）
    pred_probs = model.predict(x_resized, batch_size=test_batch_size,verbose=0)
    predictions = np.argmax(pred_probs, axis=1).tolist()
    confidences = np.max(pred_probs, axis=1).tolist()

    tf.keras.backend.clear_session()
    del model
    gc.collect()

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "predictions": predictions,   # 画像順に並んだ予測ラベルIDのリスト
        "confidences": confidences,   # 画像順に並んだ確信度のリスト
    }