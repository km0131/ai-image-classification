# [ARCHIVED 2026-07-11] mobilenet_v3 / efficientnet_lite4 の旧TensorFlow/Keras実装スナップショット。
# 移行理由・詳細は同ディレクトリの README.md を参照。ロールバック用に保管するのみで、
# このファイル自体は実行されない(importもされない)。

# ============================================================
# models_config.py 側(旧): TF依存部分
# ============================================================
#
# import tensorflow as tf
# from tensorflow.keras import mixed_precision
#
#
# def setup_precision():
#     gpus = tf.config.list_physical_devices("GPU")
#     if not gpus:
#         policy = "float32"
#     else:
#         details = tf.config.experimental.get_device_details(gpus[0])
#         cc = details.get("compute_capability", (0, 0))
#         policy = "mixed_float16" if cc[0] >= 7 else "float32"
#     mixed_precision.set_global_policy(mixed_precision.Policy(policy))
#
#
# setup_precision()
#
# MODEL_CONFIGS = {
#     'mobilenet_v3': {'framework': 'tf', 'size': (224, 224),
#                       'base': tf.keras.applications.MobileNetV3Large, 'weights': 'imagenet'},
#     'efficientnet_lite4': {'framework': 'tf', 'size': (300, 300),
#                             'base': tf.keras.applications.EfficientNetB4, 'weights': 'imagenet'},
# }

# ============================================================
# ai_logic.py 側(旧): 3モデルループ内のTF学習・エクスポートブロック
# ============================================================
#
# import tensorflow as tf
# from tensorflow.keras import layers, models, mixed_precision
#
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
# os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
#
# for model_name, config in MODEL_CONFIGS.items():
#     img_size = config["size"]
#     x_resized = np.array([cv2.resize(img, (img_size[1], img_size[0])) for img in all_x], dtype=np.float32)
#
#     if model_name == "mobilenet_v3":
#         x_resized = tf.keras.applications.mobilenet_v3.preprocess_input(x_resized)
#     elif model_name == "efficientnet_lite4":
#         x_resized = tf.keras.applications.efficientnet.preprocess_input(x_resized)
#
#     base = config["base"](input_shape=(*img_size, 3), include_top=False, weights=config.get("weights"))
#     base.trainable = False
#
#     model = models.Sequential([
#         base,
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(num_classes, activation="softmax", dtype="float32")
#     ])
#
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"]
#     )
#
#     # Phase 1: backbone凍結
#     history1 = model.fit(
#         x_resized, y_train, epochs=EPOCH1, batch_size=BATCH1, validation_split=VALIDATION,
#         callbacks=[
#             tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
#             tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7)
#         ]
#     )
#
#     # Phase 2: 末尾30%を解凍してfine-tune
#     if FINE_TUNE:
#         base.trainable = True
#         fine_tune_at = int(len(base.layers) * 0.7)
#         for layer in base.layers[:fine_tune_at]:
#             layer.trainable = False
#         for layer in base.layers:
#             if isinstance(layer, (layers.BatchNormalization, layers.LayerNormalization)):
#                 layer.trainable = False
#
#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
#             loss="sparse_categorical_crossentropy",
#             metrics=["accuracy"]
#         )
#         history2 = model.fit(
#             x_resized, y_train, epochs=EPOCH2, batch_size=BATCH2, validation_split=VALIDATION,
#             callbacks=[
#                 tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
#                 tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-8)
#             ]
#         )
#
#     # 学習曲線の統合(accuracy/loss/val_accuracy/val_lossのepoch別リスト)
#     history_dict = {k: history1.history[k] + history2.history[k] if FINE_TUNE else history1.history[k]
#                     for k in history1.history.keys()}
#
#     # 元モデル(.keras)を保存
#     native_keras_path = os.path.join(user_export_root, f"{model_name}.keras")
#     model.save(native_keras_path, save_format="keras")
#
#     # mixed_float16のままTFLiteConverterにかけると 'ERROR_NEEDS_FLEX_OPS' で失敗するため、
#     # float32ポリシーで同一アーキテクチャを再構築して重みだけコピーしてから変換する。
#     saved_weights = model.get_weights()
#     prev_policy = mixed_precision.global_policy()
#     mixed_precision.set_global_policy("float32")
#     try:
#         export_base = config["base"](input_shape=(*img_size, 3), include_top=False, weights=None)
#         export_model = models.Sequential([
#             export_base,
#             layers.GlobalAveragePooling2D(),
#             layers.Dense(num_classes, activation="softmax")
#         ])
#         export_model.set_weights(saved_weights)
#     finally:
#         mixed_precision.set_global_policy(prev_policy)
#
#     converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
#     tflite_bytes = converter.convert()
#     with open(os.path.join(export_path, "model.tflite"), "wb") as f:
#         f.write(tflite_bytes)
#
#     tf.keras.backend.clear_session()

# ============================================================
# evaluate_test_model 側(旧): TF系モデルの前処理分岐
# ============================================================
#
# if model_name == "mobilenet_v3":
#     x_resized = tf.keras.applications.mobilenet_v3.preprocess_input(x_resized)
#     apply_softmax = False  # Dense層に activation="softmax" が既に含まれるため
# elif model_name == "efficientnet_lite4":
#     x_resized = tf.keras.applications.efficientnet.preprocess_input(x_resized)
#     apply_softmax = False

# ============================================================
# main.py 側(旧): /analyze の特徴抽出器
# ============================================================
#
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.models import Model
#
# base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
# feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
#
# x = preprocess_input(np.expand_dims(cv2.resize(img_rgb, (224, 224)), axis=0))
# diversity_vector = feature_extractor.predict(x, verbose=0).flatten()[:2].tolist()
