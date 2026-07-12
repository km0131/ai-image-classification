# [ARCHIVED 2026-07-11] mobilenet_v3 / efficientnet_lite4 の旧TensorFlow/Keras実装

`tensorflow[and-cuda]` と `torch` を `requirements.txt` に同居させたところ、両者が要求する
`nvidia-cublas-cu12` 等CUDAライブラリのバージョンが競合し `pip install` が `ResolutionImpossible`
で失敗するようになったため、`mobilenet_v3` / `efficientnet_lite4` もPyTorch/timm実装
(`torch_models.py`)へ移行し、`tensorflow` を依存関係から完全に排除した。

ロールバック用に、移行直前の `ai_logic.py` / `models_config.py` のTF依存部分の実装を
`ai_logic_tf_snapshot.py` として保管する(実行には一切関与しない参照用スナップショット)。

## 旧実装の概要

- `models_config.py`: `MODEL_CONFIGS['mobilenet_v3']` は `tf.keras.applications.MobileNetV3Large`
  (224×224)、`MODEL_CONFIGS['efficientnet_lite4']` は `tf.keras.applications.EfficientNetB4`
  (300×300、※本物のEfficientNet-Lite4ではなく別アーキテクチャを誤って使っていた)。
  起動時に `setup_precision()` でGPUのTensor Core対応を見て `mixed_float16`/`float32` を切り替えていた。
- `ai_logic.py`: `Sequential([base, GlobalAveragePooling2D(), Dense(num_classes, activation="softmax")])`
  を `model.fit()` でPhase1(backbone凍結)→Phase2(末尾30%解凍してfine-tune)の2段階学習し、
  `mixed_float16` のままだと `TFLiteConverter` が `ERROR_NEEDS_FLEX_OPS` で失敗するため、
  変換直前にfloat32ポリシーで同一アーキテクチャを再構築して重みをコピーしてから
  `TFLiteConverter.from_keras_model()` で `.tflite` に変換していた。

新実装(PyTorch/timm統一後)は `torch_models.py` の `train_timm_model()` / `export_tflite()` を
`mobilenet_v3`/`efficientnet_lite4`/`mobilevit_v2` の3モデル共通で使う構成になっている。
