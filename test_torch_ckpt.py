import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

print("[INFO] Torch version:", torch.__version__)
print("[INFO] CUDA available:", torch.cuda.is_available())

# 一個簡單的模型
model = nn.Linear(8, 8)
opt = optim.AdamW(model.parameters(), lr=1e-3)

# 確保有 FAIR_RUN_DIR
run_dir = os.environ.get("FAIR_RUN_DIR", "./runs/demo")
os.makedirs(run_dir, exist_ok=True)
print("[INFO] Checkpoints will be written under:", run_dir)

# 簡單跑幾個 step
for step in range(20):
    x = torch.randn(4, 8)
    y = model(x).sum()
    y.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    print(f"[TRAIN] step {step} done")

    time.sleep(0.5)  # 放慢一點，好觀察週期 checkpoint

