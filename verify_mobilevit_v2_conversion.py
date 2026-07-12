"""
MobileViT-v2(PyTorch/timm)の .tflite 変換前後の出力比較スクリプト。

指示書Step5/6の「新旧モデルの出力比較」に相当するが、旧TF版は事前学習されていない
自作スタブ(legacy/mobilevit_v2_tf/)だったため意味のある比較対象にならない。
そのため本スクリプトでは「変換前(PyTorch) vs 変換後(.tflite)」の数値比較を行い、
ai-edge-torch変換によるAttention層の精度劣化・非対応opの有無を確認する。

使い方:
    source .venv/bin/activate
    python verify_mobilevit_v2_conversion.py
"""
import glob
import os

import cv2
import numpy as np

import mobilevit_v2_torch as mvit


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def load_sample_images(n=5):
    paths = sorted(glob.glob("./received_images/*.jpg"))[:n]
    if not paths:
        raise RuntimeError("received_images/ にサンプル画像が見つかりません")
    images = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return paths, images


def main():
    import torch

    num_classes = 5  # 比較目的のダミークラス数(分類ヘッドの重みはランダムだが、変換前後比較には影響しない)
    print(f"=== MobileViT-v2(timm mobilevitv2_100) 変換前後比較(num_classes={num_classes}) ===")

    model = mvit.build_model(num_classes=num_classes, pretrained=True)
    model.eval()
    preprocess_cfg = mvit.get_preprocess_config(model)
    print("preprocess_cfg:", preprocess_cfg)

    paths, images = load_sample_images()
    resized = np.array(
        [cv2.resize(img, (preprocess_cfg["width"], preprocess_cfg["height"])) for img in images],
        dtype=np.float32,
    )
    x = mvit.prepare_inputs(resized, preprocess_cfg)

    with torch.no_grad():
        torch_logits = model(x.to(mvit.DEVICE)).cpu().numpy()
    torch_probs = _softmax(torch_logits)

    tflite_path = "/tmp/verify_mobilevit_v2.tflite"
    mvit.export_tflite(model, preprocess_cfg, tflite_path)

    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    in_shape = input_detail["shape"]
    is_nchw = len(in_shape) == 4 and in_shape[1] == 3
    print(f"tflite入力shape: {in_shape} (NCHW={is_nchw})")

    x_np = x.numpy()
    x_input = x_np if is_nchw else np.transpose(x_np, (0, 2, 3, 1))
    x_input = x_input.astype(input_detail["dtype"])

    tflite_logits = []
    for i in range(len(x_input)):
        interpreter.set_tensor(input_detail["index"], x_input[i:i + 1])
        interpreter.invoke()
        tflite_logits.append(interpreter.get_tensor(output_detail["index"])[0])
    tflite_logits = np.array(tflite_logits)
    tflite_probs = _softmax(tflite_logits)

    max_abs_diff = np.max(np.abs(torch_probs - tflite_probs))
    mean_abs_diff = np.mean(np.abs(torch_probs - tflite_probs))
    top1_agree = np.mean(np.argmax(torch_probs, axis=1) == np.argmax(tflite_probs, axis=1))

    report_lines = [
        "MobileViT-v2 (mobilevitv2_100) 変換前後比較レポート",
        f"サンプル画像数: {len(paths)}",
        f"入力サイズ: {preprocess_cfg['height']}x{preprocess_cfg['width']}, mean={preprocess_cfg['mean']}, std={preprocess_cfg['std']}",
        f"softmax確率の最大絶対誤差: {max_abs_diff:.6f}",
        f"softmax確率の平均絶対誤差: {mean_abs_diff:.6f}",
        f"Top-1一致率(PyTorch vs TFLite): {top1_agree * 100:.1f}%",
        "",
        "画像ごとの詳細:",
    ]
    for path, tp, fp in zip(paths, torch_probs, tflite_probs):
        report_lines.append(
            f"  {os.path.basename(path)}: torch_top1={int(np.argmax(tp))}({np.max(tp):.4f}) "
            f"tflite_top1={int(np.argmax(fp))}({np.max(fp):.4f})"
        )

    report = "\n".join(report_lines)
    print("\n" + report)

    with open("mobilevit_v2_conversion_report.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\nレポートを mobilevit_v2_conversion_report.txt に保存しました")

    if max_abs_diff > 0.05:
        print("警告: 変換前後の出力差が大きいです(0.05超)。Attention層の変換精度を確認してください。")


if __name__ == "__main__":
    main()
