import os

# 3モデルとも PyTorch/timm へ統一済み。実装は torch_models.py 参照。
# 旧TF版(mobilenet_v3/efficientnet_lite4のKeras実装、mobilevit_v2の自作Attentionブロック)は
# legacy/tf_cnn_models/ 、legacy/mobilevit_v2_tf/ にアーカイブ済み。
MODEL_CONFIGS = {
    'mobilenet_v3':       {'framework': 'torch', 'size': (224, 224), 'timm_name': 'mobilenetv3_large_100'},
    # 'tf_efficientnet_lite4' のネイティブ解像度は380x380(旧コードは別アーキテクチャの
    # EfficientNetB4を誤って"efficientnet_lite4"として使っており300x300だった)。
    'efficientnet_lite4': {'framework': 'torch', 'size': (380, 380), 'timm_name': 'tf_efficientnet_lite4'},
    'mobilevit_v2':       {'framework': 'torch', 'size': (256, 256), 'timm_name': 'mobilevitv2_100'},
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
