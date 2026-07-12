"""
[ARCHIVED 2026-07-10] 旧TensorFlow版 MobileViT-v2。

models_config.py から移動。実在するMobileViT-v2アーキテクチャではなく、
Conv+LayerNorm+MultiHeadAttentionを数層組み合わせただけの自作モデルで、
`./weights/mobilevitv2_imagenet.keras` が存在しない限り事前学習済み重みも
読み込まれない(=実質ランダム初期化のまま使われていた)。

移行理由・詳細は README.md の「MobileViT-v2移行について」を参照。
ロールバック用に削除はせずここに保管する。
"""
import os
import tensorflow as tf
from tensorflow.keras import layers, models


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
