# client/train_gpu_minimal.py
import os, sys, time, signal, random
import numpy as np
import torch
import torch.nn as nn

# ------------ 基本設定 ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    print("[WARN] No CUDA GPU found. This script is meant to run on a GPU.", flush=True)
    # 仍可在 CPU 跑測流程，但不建議。若要強制中止改成 sys.exit(11)

# 由排程器提供（也可自己給）
CKPT_PATH = os.getenv("FAIR_CKPT_PATH", os.path.expanduser("~/ckpts/minimal.ckpt"))
USER_ID   = os.getenv("FAIR_USER_ID", "uX")
JOB_ID    = os.getenv("FAIR_JOB_ID",  "JX")

# ------------ 訊號處理 ------------
_should_ckpt = False
def _on_sigusr1(sig, frame):
    global _should_ckpt
    _should_ckpt = True
signal.signal(signal.SIGUSR1, _on_sigusr1)

# ------------ 小模型 + 假資料 ------------
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256),  nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x): return self.net(x)

model = TinyNet().to(device)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()

BATCH = 512
STEPS_PER_EPOCH = 2000  # 一個 epoch 內有多少 step（用假資料不影響）
MAX_EPOCHS = 100000     # 實際上時間片會提前結束

# ------------ 還原（若有 ckpt） ------------
def save_ckpt(epoch, step):
    tmp = CKPT_PATH + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optim": opt.state_dict(),
        "epoch": epoch,
        "step": step + 1,  # 下一個要跑的 batch
        
        "rng_cpu": torch.random.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        "rng_py": random.getstate(),
        "rng_np": np.random.get_state(),

        
    }, tmp)
    os.replace(tmp, CKPT_PATH)

'''

def load_ckpt():
    if not os.path.exists(CKPT_PATH):
        return 0, 0
    ck = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optim"])
    torch.random.set_rng_state(ck["rng_cpu"])
    if ck.get("rng_cuda") is not None and device.type == "cuda":
        torch.cuda.set_rng_state_all(ck["rng_cuda"])
    random.setstate(ck["rng_py"])
    np.random.set_state(ck["rng_np"])
    return ck["epoch"], ck["step"]
    
'''

def _to_byte_tensor(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.uint8, device='cpu')
    # 可能是 list / bytes / numpy array 等
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.uint8)
    except Exception:
        pass
    if isinstance(x, (bytes, bytearray)):
        return torch.tensor(list(x), dtype=torch.uint8)
    if isinstance(x, list):
        return torch.tensor(x, dtype=torch.uint8)
    return None

def load_ckpt():
    if not os.path.exists(CKPT_PATH):
        return 0, 0
    ck = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optim"])

    # RNG 還原：容錯處理
    try:
        rcpu = _to_byte_tensor(ck.get("rng_cpu"))
        if rcpu is not None:
            torch.random.set_rng_state(rcpu)
    except Exception as e:
        print(f"[WARN] skip restoring CPU RNG: {e}", flush=True)

    try:
        rcuda = ck.get("rng_cuda")
        if rcuda is not None and device.type == "cuda":
            # rcuda 可能是 list of tensors / list of lists
            if isinstance(rcuda, list):
                fixed = []
                for s in rcuda:
                    t = _to_byte_tensor(s)
                    if t is not None:
                        fixed.append(t)
                if fixed:
                    torch.cuda.set_rng_state_all(fixed)
    except Exception as e:
        print(f"[WARN] skip restoring CUDA RNG: {e}", flush=True)

    try:
        random.setstate(ck["rng_py"])
    except Exception as e:
        print(f"[WARN] skip restoring PY RNG: {e}", flush=True)
    try:
        np.random.set_state(ck["rng_np"])
    except Exception as e:
        print(f"[WARN] skip restoring NP RNG: {e}", flush=True)

    return ck.get("epoch", 0), ck.get("step", 0)


start_epoch, start_step = load_ckpt()

print(f"[{USER_ID}/{JOB_ID}] start pid={os.getpid()} device={device} from (epoch={start_epoch}, step={start_step})", flush=True)

# ------------ 訓練 Loop（用隨機資料，但會吃到 GPU） ------------
for epoch in range(start_epoch, MAX_EPOCHS):
    for step in range(start_step, STEPS_PER_EPOCH):
        # 產生隨機 batch（會觸發 GPU 計算）
        x = torch.randn(BATCH, 1024, device=device)
        y = torch.randint(0, 10, (BATCH,), device=device)

        # 前向/反向/更新
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = lossf(logits, y)
        loss.backward()
        opt.step()

        # 每隔一點點印一下，方便觀察
        if step % 200 == 0:
            print(f"[{USER_ID}/{JOB_ID}] epoch={epoch} step={step} loss={loss.item():.4f}", flush=True)

        # 收到換片訊號 → 在「batch 結束」存檔並正常退出
        if _should_ckpt:
            if device.type == "cuda":
                torch.cuda.synchronize()
            save_ckpt(epoch, step)
            print(f"[{USER_ID}/{JOB_ID}] got SIGUSR1, checkpoint saved → exit(0)", flush=True)
            sys.exit(0)  # 0 = 正常換片退出

    # 一個 epoch 結束後，下一輪從 step=0 開始
    start_step = 0

# 如果真的自然跑完（理論上用不到）
sys.exit(10)  # 10 = 任務自然完成

