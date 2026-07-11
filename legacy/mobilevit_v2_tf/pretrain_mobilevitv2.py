# [ARCHIVED 2026-07-10] 旧TensorFlow版 MobileViT-v2 の事前学習スクリプト。
# tf.keras.datasets.imagenet.load_data() はTF/Kerasに存在しないAPIであり、
# このスクリプトは移行前から実行不能なスタブだった。ロールバック用に削除せず保管する。
# 詳細は README.md の「MobileViT-v2移行について」を参照。
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# ここは現在作成したMobileViTv2をimport
from model import MobileViTv2

print("GPU一覧:")
print(tf.config.list_physical_devices("GPU"))


IMG_SIZE = 256
NUM_CLASSES = 1000

SAVE_PATH = "./weights/mobilevitv2_imagenet.keras"


# =========================
# ImageNetデータセット
# =========================

(x_train, y_train), (x_val, y_val) = tf.keras.datasets.imagenet.load_data(
    label_mode="fine"
)


x_train = tf.image.resize(
    x_train,
    (IMG_SIZE, IMG_SIZE)
)

x_val = tf.image.resize(
    x_val,
    (IMG_SIZE, IMG_SIZE)
)


x_train = tf.cast(x_train, tf.float32) / 255.0
x_val = tf.cast(x_val, tf.float32) / 255.0


# =========================
# MobileViTv2 Backbone
# =========================

base = MobileViTv2(
    input_shape=(256,256,3),
    include_top=False,
    weights=None
)


model = models.Sequential([

    base,

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )

])


model.summary()


# =========================
# 学習設定
# =========================

model.compile(

    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=1e-5
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


callbacks=[

    tf.keras.callbacks.ModelCheckpoint(
        SAVE_PATH,
        save_weights_only=True,
        save_best_only=True,
        monitor="val_accuracy"
    ),

    tf.keras.callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True
    )

]


# =========================
# 学習
# =========================

model.fit(

    x_train,
    y_train,

    validation_data=(
        x_val,
        y_val
    ),

    epochs=100,

    batch_size=64,

    callbacks=callbacks

)


print(
    "保存完了:",
    SAVE_PATH
)