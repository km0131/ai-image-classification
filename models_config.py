import os
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision


def MobileViTv2(input_shape=(256, 256, 3), include_top=False, weights=None):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="swish")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="swish")(x)

    shortcut = layers.Conv2D(96, 3, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(shortcut)

    h = input_shape[0] // 4
    w = input_shape[1] // 4
    x = layers.Reshape((-1, 96))(x)

    attention = layers.MultiHeadAttention(num_heads=4, key_dim=24)
    x = attention(x, x)
    x = layers.Reshape((h, w, 96))(x)
    x = layers.Add()([shortcut, x])
    x = layers.Conv2D(128, 3, padding="same", activation="swish")(x)

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)

    model = models.Model(inputs, x, name="MobileViTv2")

    if weights is not None:
        if os.path.exists(weights):
            print("Loading ImageNet pretrained weight:", weights)
            model.load_weights(weights, skip_mismatch=True)
            print("MobileViTv2 ImageNet weight loaded")
        else:
            print("Weight file not found:", weights)
    return model


def setup_precision():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("=============================================================")
        print("GPUが検出されませんでした。CPU実行のため float32 を使用します")
        print("=============================================================")
        policy = "float32"
    else:
        details = tf.config.experimental.get_device_details(gpus[0])
        cc = details.get("compute_capability", (0, 0))
        if cc[0] >= 7:
            print("=============================================================")
            print(f"Tensor Core対応GPUを検出 ({details.get('device_name', '不明')})。mixed_float16 を有効化します")
            print("=============================================================")
            policy = "mixed_float16"
        else:
            policy = "float32"
    mixed_precision.set_global_policy(mixed_precision.Policy(policy))


# 起動時初期化
setup_precision()

MODEL_CONFIGS = {
    'mobilenet_v3': {'size': (224, 224), 'base': tf.keras.applications.MobileNetV3Large, 'weights': 'imagenet'},
    'efficientnet_lite4': {'size': (300, 300), 'base': tf.keras.applications.EfficientNetB4, 'weights': 'imagenet'},
    'mobilevit_v2': {'size': (256, 256), 'base': MobileViTv2, 'weights': './weights/mobilevitv2_imagenet.keras'}
}

# モードの切り替え
AI_MODE = os.getenv("AI_MODE", "production")
if AI_MODE == "test":
    EPOCH1, EPOCH2, BATCH1, BATCH2, VALIDATION, FINE_TUNE = 1, 1, 2, 2, 0.1, False
    print("=============================================================")
    print("テストモード")
    print("=============================================================")
else:
    EPOCH1, EPOCH2, BATCH1, BATCH2, VALIDATION, FINE_TUNE = 50, 30, 32, 16, 0.2, True
    print("=============================================================")
    print("本番モード")
    print("=============================================================")
