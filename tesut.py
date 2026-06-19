import tensorflow as tf

# 使用可能なGPUデバイスの一覧を表示
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n見つかったGPUの数: {len(gpus)}")
    for gpu in gpus:
        print(f"GPUデバイス名: {gpu}")
    # 実際に計算がGPUで行われるか簡単な演算で確認
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0]])
        b = tf.constant([[4.0, 5.0, 6.0]])
        c = tf.matmul(a, b, transpose_b=True)
        print("GPUでの計算結果:", c.numpy())
else:
    print("GPUが見つかりません。")