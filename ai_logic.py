import os
import gc
import json
import shutil
import zipfile
import cv2
import numpy as np
import requests
import torch

if os.getenv("FORCE_CPU_FOR_TEST", "false").lower() == "true":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from models_config import MODEL_CONFIGS, EPOCH1, EPOCH2, BATCH1, BATCH2, VALIDATION, FINE_TUNE
import torch_models as tm


def _train_and_export_torch_model(model_name, config, x_resized, y_train, num_classes, user_export_root):
    """PyTorch(timm)での学習・エクスポート。3モデル(mobilenet_v3/efficientnet_lite4/mobilevit_v2)共通。

    出力(user_export_root配下):
      {model_name}/model.tflite  … フロント配信用(toPublicURLがこのパスを指す)
      {model_name}.tflite        … Go /test 評価用(トップレベル単体ファイル規約)
      {model_name}.pt            … 元モデルのアーカイブ(変換済みモデルと元モデルの両方を返す要件のため)
    """
    model, history, preprocess_cfg = tm.train_timm_model(
        x_resized, y_train, num_classes,
        epoch1=EPOCH1, epoch2=EPOCH2, batch1=BATCH1, batch2=BATCH2,
        validation_split=VALIDATION, fine_tune=FINE_TUNE,
        timm_name=config["timm_name"],
    )

    expected_h, expected_w = config["size"]
    if (preprocess_cfg["height"], preprocess_cfg["width"]) != (expected_h, expected_w):
        print(f"[{model_name}] 警告: timmの既定入力サイズ({preprocess_cfg['height']}x{preprocess_cfg['width']}) が "
              f"models_config.pyのsize({expected_h}x{expected_w})と異なります。models_config.pyのsizeを合わせてください。")

    epoch_accuracies = [float(v) for v in history["accuracy"]]
    epoch_losses = [float(v) for v in history["loss"]]
    val_accuracies = [float(v) for v in history["val_accuracy"]]
    val_losses = [float(v) for v in history["val_loss"]]

    curve = [
        {"epoch": i + 1, "accuracy": epoch_accuracies[i], "loss": epoch_losses[i],
         "val_accuracy": val_accuracies[i], "val_loss": val_losses[i]}
        for i in range(len(epoch_accuracies))
    ]
    best_epoch = int(np.argmin(val_losses))
    summary_entry = {"accuracy": val_accuracies[best_epoch], "loss": val_losses[best_epoch]}

    export_path = os.path.join(user_export_root, model_name)
    os.makedirs(export_path, exist_ok=True)

    tflite_path = os.path.join(export_path, "model.tflite")
    tm.export_tflite(model, preprocess_cfg, tflite_path, model_label=model_name)
    shutil.copyfile(tflite_path, os.path.join(user_export_root, f"{model_name}.tflite"))
    tm.export_pt(model, os.path.join(user_export_root, f"{model_name}.pt"), model_label=model_name)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return curve, summary_entry, val_losses[best_epoch]


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

    # 3モデルループ(全てPyTorch/timm経由)
    for model_name, config in MODEL_CONFIGS.items():
        print(f"=== Training: {model_name} ===")
        img_size = config["size"]
        x_resized = np.array([cv2.resize(img, (img_size[1], img_size[0])) for img in all_x], dtype=np.float32)

        curve, summary_entry, best_val_loss = _train_and_export_torch_model(
            model_name, config, x_resized, y_train, num_classes, user_export_root
        )
        all_models_curves[model_name] = curve
        model_summary[model_name] = summary_entry
        summary_loss = best_val_loss
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


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _preprocess_for_tflite_eval(model_name, config, all_test_x):
    """TFLite評価用の前処理。学習時(run_training_and_callback / _train_and_export_torch_model)と
    必ず同じ前処理を適用すること。3モデル共通でtimmのmean/stdに基づく正規化を行う。
    timmの分類ヘッドはどのモデルも生ロジットを返すため、apply_softmaxは常にTrue。
    戻り値: (NHWC float32配列, apply_softmax)
    """
    # 重みのダウンロードは不要(pretrained=False)。timmが持つmean/std/入力サイズのメタデータだけを参照する。
    ref_model = tm.build_model(num_classes=1, timm_name=config["timm_name"], pretrained=False)
    preprocess_cfg = tm.get_preprocess_config(ref_model)
    del ref_model

    img_h, img_w = preprocess_cfg["height"], preprocess_cfg["width"]
    x_resized = np.array([cv2.resize(img, (img_w, img_h)) for img in all_test_x], dtype=np.float32)
    mean = np.array(preprocess_cfg["mean"], dtype=np.float32)
    std = np.array(preprocess_cfg["std"], dtype=np.float32)
    x_norm = (x_resized / 255.0 - mean) / std
    return x_norm, True


def _run_tflite_interpreter(model_path, x_hwc, apply_softmax):
    """共通のTFLite Interpreter実行ループ。x_hwcはNHWC(前処理済み)。
    tf.lite.InterpreterはTF 2.20で削除予定のため、後継のai_edge_litert.Interpreterを使う。
    変換後の入力レイアウトがNCHW/NHWCどちらになるか断定せず、shapeから判定して転置する
    (litert-torch由来はNCHWになりやすいが、実際のモデルshapeを信頼する)。"""
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    in_shape = input_detail["shape"]
    is_nchw = len(in_shape) == 4 and in_shape[1] == 3
    x_input = np.transpose(x_hwc, (0, 3, 1, 2)) if is_nchw else x_hwc
    x_input = x_input.astype(input_detail["dtype"])

    all_probs = []
    for i in range(len(x_input)):
        interpreter.set_tensor(input_detail["index"], x_input[i:i + 1])
        interpreter.invoke()
        out = interpreter.get_tensor(output_detail["index"])[0]
        all_probs.append(_softmax(out) if apply_softmax else out)
    return np.array(all_probs)


# Goから届いた .tflite モデルと画像ZIPを使って性能テストをするロジック(3モデル共通)
def evaluate_test_model(model_path, extract_dir, model_name, all_test_x, all_test_y):
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        return {"error": f"Unknown model_name: {model_name}"}

    x_pre, apply_softmax = _preprocess_for_tflite_eval(model_name, config, all_test_x)
    y_test = np.array(all_test_y, dtype=np.int64)

    print(f"[{model_name}] evaluate開始(TFLite Interpreter): 画像枚数={len(all_test_x)}")

    all_probs = _run_tflite_interpreter(model_path, x_pre, apply_softmax)
    predictions = np.argmax(all_probs, axis=1)
    confidences = np.max(all_probs, axis=1)
    losses = [-np.log(max(all_probs[i][y_test[i]], 1e-9)) for i in range(len(y_test))]

    accuracy = float(np.mean(predictions == y_test)) if len(y_test) else 0.0
    loss = float(np.mean(losses)) if losses else 0.0

    return {
        "loss": loss,
        "accuracy": accuracy,
        "predictions": predictions.tolist(),   # 画像順に並んだ予測ラベルIDのリスト
        "confidences": confidences.tolist(),   # 画像順に並んだ確信度のリスト
    }
