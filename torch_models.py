"""
PyTorch / timm による3モデル(mobilenet_v3 / efficientnet_lite4 / mobilevit_v2)共通の
学習・前処理・変換モジュール。timm_name を差し替えるだけでどのモデルにも使える汎用実装。

旧TensorFlow版(legacy/mobilevit_v2_tf/, legacy/tf_cnn_models/)からの移行に伴い、
「Phase1: backbone凍結 → Phase2: 末尾30%を解凍してfine-tune」という2フェーズ構成を
PyTorchで再現し、ai_logic.py 側の学習曲線集計ロジックがそのまま使える形式
(accuracy/loss/val_accuracy/val_lossのepoch別リスト)で履歴を返す。
"""
import copy
import os

import numpy as np
import timm
import torch
import torch.nn as nn
from timm.data import resolve_data_config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_precision_torch() -> bool:
    """Tensor Core対応GPU(compute capability >= 7.0)ならAMP(fp16)を使う。models_config.setup_precisionのPyTorch版。"""
    if not torch.cuda.is_available():
        print("==================================================================================================")
        print("[torch] GPUが検出されませんでした。CPU実行のためfloat32を使用します")
        print("==================================================================================================")
        return False
    major, _ = torch.cuda.get_device_capability(0)
    if major >= 7:
        print("==================================================================================================")
        print(f"[torch] Tensor Core対応GPUを検出 ({torch.cuda.get_device_name(0)})。AMP(float16)を有効化します")
        print("==================================================================================================")
        return True
    return False


USE_AMP = setup_precision_torch()


def _patch_linear_self_attention_for_litert():
    """timmのLinearSelfAttention._forward_self_attn は `key * context_scores` で
    [B,d,P,N] * [B,1,P,N] というチャンネル次元(非末尾次元)のブロードキャスト乗算を行うが、
    litert-torch(旧ai-edge-torch)の変換パスはこの形のブロードキャストを 'tfl.mul' に
    legalizeできず `operands don't have broadcast-compatible shapes` で変換が失敗する
    (verify_mobilevit_v2_conversion.py で実際に確認済み)。
    数学的に同一の `context_scores.expand_as(key)` を先に計算しておくことで、
    形状が完全一致した乗算にしてこの非対応opを回避する。
    """
    import torch.nn.functional as F
    from timm.models.mobilevit import LinearSelfAttention

    if getattr(LinearSelfAttention, "_litert_patched", False):
        return

    def _forward_self_attn(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(x)
        query, key, value = qkv.split([1, self.embed_dim, self.embed_dim], dim=1)
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_drop(context_scores)
        context_vector = (key * context_scores.expand_as(key)).sum(dim=-1, keepdim=True)
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        out = self.out_drop(out)
        return out

    def _forward_cross_attn(self, x: torch.Tensor, x_prev=None) -> torch.Tensor:
        batch_size, in_dim, kv_patch_area, kv_num_patches = x.shape
        q_patch_area, q_num_patches = x.shape[-2:]
        assert kv_patch_area == q_patch_area
        qk = F.conv2d(x_prev, weight=self.qkv_proj.weight[:self.embed_dim + 1],
                       bias=self.qkv_proj.bias[:self.embed_dim + 1])
        query, key = qk.split([1, self.embed_dim], dim=1)
        value = F.conv2d(
            x, weight=self.qkv_proj.weight[self.embed_dim + 1],
            bias=self.qkv_proj.bias[self.embed_dim + 1] if self.qkv_proj.bias is not None else None,
        )
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_drop(context_scores)
        context_vector = (key * context_scores.expand_as(key)).sum(dim=-1, keepdim=True)
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        out = self.out_drop(out)
        return out

    LinearSelfAttention._forward_self_attn = _forward_self_attn
    LinearSelfAttention._forward_cross_attn = _forward_cross_attn
    LinearSelfAttention._litert_patched = True


_patch_linear_self_attention_for_litert()


def build_model(num_classes: int, timm_name: str = "mobilevitv2_100", pretrained: bool = True) -> nn.Module:
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    return model.to(DEVICE)


def get_preprocess_config(model: nn.Module) -> dict:
    """timm既定の前処理設定(入力サイズ・mean・std)を返す。ハードコードしない(指示書の制約事項参照)。"""
    cfg = resolve_data_config({}, model=model)
    input_size = cfg["input_size"]  # (C, H, W)
    return {
        "height": input_size[1],
        "width": input_size[2],
        "mean": list(cfg["mean"]),
        "std": list(cfg["std"]),
    }


def prepare_inputs(x_uint8_nhwc: np.ndarray, preprocess_cfg: dict) -> torch.Tensor:
    """0-255のNHWC numpy配列(RGB, 既にpreprocess_cfgのサイズにリサイズ済み)をtimm既定の正規化済みNCHW Tensorに変換する。"""
    x = torch.from_numpy(x_uint8_nhwc).float() / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
    mean = torch.tensor(preprocess_cfg["mean"]).view(1, -1, 1, 1)
    std = torch.tensor(preprocess_cfg["std"]).view(1, -1, 1, 1)
    return (x - mean) / std


class _EarlyStopping:
    """Keras EarlyStopping(monitor="val_loss", restore_best_weights=True)相当。"""

    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("inf")
        self.best_state = None
        self.wait = 0

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best:
            self.best = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _leaf_modules(model: nn.Module):
    return [m for m in model.modules() if len(list(m.children())) == 0]


def _freeze_backbone(model: nn.Module):
    """Phase1: 分類ヘッド以外を全凍結(TF版 base.trainable=False 相当)。"""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def _unfreeze_tail(model: nn.Module, ratio: float = 0.7):
    """Phase2: 末尾(1-ratio)を解凍してfine-tune。BatchNorm/LayerNormは凍結のまま(TF版と同じ方針)。"""
    leaves = _leaf_modules(model)
    n = len(leaves)
    fine_tune_at = int(n * ratio)
    frozen_norm_modules = []
    for i, m in enumerate(leaves):
        trainable = i >= fine_tune_at
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
            trainable = False
            frozen_norm_modules.append(m)
        for p in m.parameters(recurse=False):
            p.requires_grad = trainable
    for p in model.get_classifier().parameters():
        p.requires_grad = True
    return frozen_norm_modules


class _FrozenNormEval(nn.Module):
    """forward前に指定モジュールをeval()化するだけの薄いラッパー(凍結BN/LNの移動統計を更新させないため)。"""

    def __init__(self, model: nn.Module, frozen_norm_modules):
        super().__init__()
        self.model = model
        self.frozen_norm_modules = frozen_norm_modules

    def forward(self, x):
        self.model.train()
        for m in self.frozen_norm_modules:
            m.eval()
        return self.model(x)


def _run_epochs(
    model: nn.Module,
    x_tr: torch.Tensor, y_tr: torch.Tensor,
    x_val: torch.Tensor, y_val: torch.Tensor,
    epochs: int, batch_size: int, lr: float,
    reduce_lr_patience: int, reduce_lr_min_lr: float,
    early_stop_patience: int,
    forward_fn,
) -> dict:
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=reduce_lr_patience, min_lr=reduce_lr_min_lr
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    criterion = nn.CrossEntropyLoss()
    early_stopping = _EarlyStopping(patience=early_stop_patience)

    n = x_tr.shape[0]
    history = {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        running_loss, running_correct = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = x_tr[idx].to(DEVICE), y_tr[idx].to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = forward_fn(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * xb.size(0)
            running_correct += (logits.argmax(dim=1) == yb).sum().item()

        train_loss = running_loss / n
        train_acc = running_correct / n

        val_loss, val_acc = _evaluate(model, x_val, y_val, batch_size, criterion, forward_fn)

        history["accuracy"].append(train_acc)
        history["loss"].append(train_loss)
        history["val_accuracy"].append(val_acc)
        history["val_loss"].append(val_loss)

        print(f"epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} "
              f"- val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

        scheduler.step(val_loss)
        if early_stopping.step(val_loss, model):
            print(f"EarlyStopping: patience({early_stop_patience})に達したため打ち切ります")
            break

    early_stopping.restore(model)
    return history


@torch.no_grad()
def _evaluate(model: nn.Module, x_val: torch.Tensor, y_val: torch.Tensor, batch_size: int, criterion, forward_fn):
    if x_val.shape[0] == 0:
        return 0.0, 0.0
    model.eval()
    n = x_val.shape[0]
    total_loss, total_correct = 0.0, 0
    for start in range(0, n, batch_size):
        xb = x_val[start:start + batch_size].to(DEVICE)
        yb = y_val[start:start + batch_size].to(DEVICE)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = forward_fn(xb)
            loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
    return total_loss / n, total_correct / n


def train_timm_model(
    x_resized: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    epoch1: int, epoch2: int, batch1: int, batch2: int,
    validation_split: float, fine_tune: bool,
    timm_name: str,
):
    """
    ai_logic._train_and_export_torch_model から呼ばれるエントリポイント。
    3モデル(mobilenet_v3/efficientnet_lite4/mobilevit_v2)共通で使う汎用関数。
    x_resized: すでにpreprocess_cfgの入力サイズにcv2.resize済みのNHWC uint8/float配列(0-255, RGB)
    戻り値: (model, history_dict, preprocess_cfg)
      history_dict は旧TF版と同じキー("accuracy"/"loss"/"val_accuracy"/"val_loss")のepoch別リストで、
      Phase1+Phase2(fine_tune時)を結合済み。
    """
    model = build_model(num_classes, timm_name=timm_name, pretrained=True)
    preprocess_cfg = get_preprocess_config(model)

    x_all = prepare_inputs(x_resized, preprocess_cfg)
    y_all = torch.from_numpy(y_train.astype(np.int64))

    n = x_all.shape[0]
    n_val = int(n * validation_split)
    if n_val > 0:
        x_tr, y_tr = x_all[:-n_val], y_all[:-n_val]
        x_val, y_val = x_all[-n_val:], y_all[-n_val:]
    else:
        x_tr, y_tr = x_all, y_all
        x_val, y_val = x_all[:0], y_all[:0]

    # Phase1: backbone凍結、分類ヘッドのみ学習
    _freeze_backbone(model)
    history1 = _run_epochs(
        model, x_tr, y_tr, x_val, y_val,
        epochs=epoch1, batch_size=batch1, lr=1e-4,
        reduce_lr_patience=3, reduce_lr_min_lr=1e-7, early_stop_patience=5,
        forward_fn=model,
    )

    history = history1
    if fine_tune:
        # Phase2: 末尾30%を解凍してfine-tune(BN/LNは凍結のまま)
        frozen_norm_modules = _unfreeze_tail(model, ratio=0.7)
        wrapped = _FrozenNormEval(model, frozen_norm_modules)
        history2 = _run_epochs(
            model, x_tr, y_tr, x_val, y_val,
            epochs=epoch2, batch_size=batch2, lr=1e-6,
            reduce_lr_patience=3, reduce_lr_min_lr=1e-8, early_stop_patience=5,
            forward_fn=wrapped,
        )
        history = {k: history1[k] + history2[k] for k in history1}

    model.eval()
    return model, history, preprocess_cfg


def export_pt(model: nn.Module, out_path: str, model_label: str = "model"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"[{model_label}] PyTorch重みを保存しました: {out_path}")


def export_tflite(model: nn.Module, preprocess_cfg: dict, out_path: str, model_label: str = "model"):
    """litert-torch(旧ai-edge-torch。2026年にlitert-torchへ改称・移行済み)で.tfliteへ変換する。
    Attention層(mobilevit_v2)の変換失敗/精度劣化に注意
    (必ずverify_mobilevit_v2_conversion.pyで変換前後比較を行うこと。CNN系は該当リスクなし)。"""
    import litert_torch

    model = model.eval().to("cpu")
    sample_input = torch.randn(1, 3, preprocess_cfg["height"], preprocess_cfg["width"])
    edge_model = litert_torch.convert(model, (sample_input,))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    edge_model.export(out_path)
    model.to(DEVICE)
    print(f"[{model_label}] .tfliteへの変換成功: {out_path}")
