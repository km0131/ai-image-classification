import torch

# 使用可能なGPUデバイスの一覧を表示
if torch.cuda.is_available():
    count = torch.cuda.device_count()
    print(f"\n見つかったGPUの数: {count}")
    for i in range(count):
        print(f"GPUデバイス名: {torch.cuda.get_device_name(i)}")
    # 実際に計算がGPUで行われるか簡単な演算で確認
    a = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    b = torch.tensor([[4.0, 5.0, 6.0]], device="cuda")
    c = torch.matmul(a, b.transpose(0, 1))
    print("GPUでの計算結果:", c.cpu().numpy())
else:
    print("GPUが見つかりません。")
