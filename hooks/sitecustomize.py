import os, time, json, signal, random, shutil, gc, weakref, types
import torch
import torch.nn as nn
import torch.optim as optim
import inspect

_RUN_DIR = os.environ.get("FAIR_RUN_DIR", "./runs/exp1")
_PERIOD_SEC = int(os.environ.get("FAIR_CKPT_PERIOD_SEC", "600"))
_PERIOD_STEPS = int(os.environ.get("FAIR_CKPT_PERIOD_STEPS", "0"))
_T_FULL = int(os.environ.get("T_FULL_SEC", "40"))
_SAFETY = int(os.environ.get("CKPT_SAFETY_SEC", "5"))
_WARN_SEC = int(os.environ.get("WARN_SEC", "120"))

os.makedirs(_RUN_DIR, exist_ok=True)
print("[fair] sitecustomize.py loaded!!!!", flush=True)

class _ABWriter:
    # creater dirs ckpt_slotA, ckpt_slotB and tmp
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.slot_names = ["ckpt_slotA","ckpt_slotB"]
        for s in self.slot_names:
            os.makedirs(os.path.join(run_dir, s), exist_ok=True)
        self.tmp = os.path.join(run_dir, "tmp"); os.makedirs(self.tmp, exist_ok=True)
    # 從 LATEST 檔案中讀取上次的進度是存在 ckpt_slotA 還是 ckpt_slotB 並回傳
    def _active(self):
        p = os.path.join(self.run_dir, "LATEST")
        if not os.path.exists(p): return self.slot_names[0]
        with open(p,"r") as f: cur=f.read().strip()
        return cur if cur in self.slot_names else self.slot_names[0]
    # 獲取下一次要寫檔在 ckpt_slotA 還是 ckpt_slotB
    def _next(self):
        return self.slot_names[1] if self._active()==self.slot_names[0] else self.slot_names[0]
    # 安全地將 data 寫入 path 檔案
    def _atomic_bytes(self, path, data:bytes):
        d=os.path.dirname(path); os.makedirs(d, exist_ok=True)
        tmp=os.path.join(self.tmp, f".tmp_{int(time.time()*1e6)}_{random.randint(0,1<<16)}")
        with open(tmp,"wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    # 安全地把一個 Python 物件存成 JSON 檔案
    def _atomic_json(self, path, obj):
        self._atomic_bytes(path, json.dumps(obj, ensure_ascii=False).encode("utf-8"))
    # 將 src 複製到 dst 
    def _atomic_copy(self, src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp=os.path.join(self.tmp, f".tmp_{int(time.time()*1e6)}_{random.randint(0,1<<16)}")
        with open(src,"rb") as s, open(tmp,"wb") as t:
            shutil.copyfileobj(s,t, length=1<<20); t.flush(); os.fsync(t.fileno())
        os.replace(tmp,dst)
    # 將 slot 字串原子寫入 LATEST 檔案
    def write_latest(self, slot):
        latest=os.path.join(self.run_dir,"LATEST"); tmp=latest+".tmp"
        with open(tmp,"w") as f:
            f.write(slot); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, latest)
    # 把一個 PyTorch 物件 (obj) 安全地存到 fname
    def save_state(self, slot, fname, obj):
        tmpf=os.path.join(self.tmp, f".state_{int(time.time()*1e6)}_{random.randint(0,1<<16)}.pt")
        torch.save(obj, tmpf)
        self._atomic_copy(tmpf, os.path.join(self.run_dir, slot, fname))
        try: os.remove(tmpf)
        except: pass
    # 呼叫 _atomic_json 將 obj 物件存成 JSON 檔案
    def save_json(self, slot, fname, obj):
        self._atomic_json(os.path.join(self.run_dir, slot, fname), obj)
    def finalize(self, slot):
        self.write_latest(slot)

_writer = _ABWriter(_RUN_DIR)

_last_full_ts = 0.0
_last_full_step = 0
_deadline = None
_pending_warn = False
_term_pending = False
_global_step = 0
_last_success_ckpt_ts = 0
_last_seen_scaler = weakref.WeakValueDictionary()
_param_to_model = weakref.WeakKeyDictionary()
_optimizer_to_model = weakref.WeakKeyDictionary()
_optimizer_to_scheduler = weakref.WeakKeyDictionary()

# 猴補 Module.__init__ → 捕捉模型 & 參數
_orig_module_init = nn.Module.__init__
def _patched_module_init(self, *args, **kwargs):
    self._fair_init_args = {"args": args, "kwargs": kwargs}
    print(f"[DEBUG] Caught model init: {self.__class__.__name__}, args={args}, kwargs={kwargs}", flush=True)
    _orig_module_init(self, *args, **kwargs)
    for p in self.parameters(recurse=True):
        _param_to_model[p] = self
nn.Module.__init__ = _patched_module_init

# 猴補 LRScheduler.__init__ → 建立 optimizer→scheduler 對應
_orig_sched_init = optim.lr_scheduler._LRScheduler.__init__
def _patched_sched_init(self, optimizer, *args, **kwargs):
    _orig_sched_init(self, optimizer, *args, **kwargs)
    _optimizer_to_scheduler[optimizer] = self
optim.lr_scheduler._LRScheduler.__init__ = _patched_sched_init

# 查詢函式
def _discover_model_from_optimizer(opt):
    return _optimizer_to_model.get(opt, None)

def _get_scheduler_for_optimizer(opt):
    return _optimizer_to_scheduler.get(opt, None)

# 打包要存的資訊
def _pack_full(opt, reason="normal"):
    model = _discover_model_from_optimizer(opt)
    scaler = None
    for k,v in list(_last_seen_scaler.items()): scaler=v
    scheduler = _get_scheduler_for_optimizer(opt)
    dl_state = None
    ema_state = None
    meta = {"ts": int(time.time()), "level":"full", "trigger": reason, "reason":"none", "global_step": int(_global_step), "lr_step": int(getattr(scheduler,'last_epoch',0)), "rank": 0, "last_success_ckpt_ts": int(_last_success_ckpt_ts)}
    return {"model": (model.state_dict() if model is not None else None),
            "optimizer": opt.state_dict(),
            "scaler": (scaler.state_dict() if scaler is not None else None),
            "scheduler": (scheduler.state_dict() if scheduler is not None else None),
            "dataloader": dl_state,
            "ema": ema_state,
            "meta": meta}, meta

# 存完整資訊
def _write_full(opt, reason):
    global _last_success_ckpt_ts
    slot = _writer._next()
    state, meta = _pack_full(opt, reason)
    _writer.save_state(slot, "rank0.ckpt", state)
    _writer.save_json(slot, "meta.json", state["meta"])
    _writer.finalize(slot)
    _last_success_ckpt_ts = int(time.time())
    return "full"

# 存部份資訊
def _write_stub(reason="timeout_during_flush"):
    slot = _writer._next()
    meta = {
        "ts": int(time.time()), 
        "level":"stub", 
        "trigger":"warn_flush", 
        "reason": reason, 
        "global_step": int(_global_step), 
        "lr_step": 0, 
        "rank": 0, 
        "last_success_ckpt_ts": int(_last_success_ckpt_ts)
    }
    _writer.save_json(slot, "meta.json", meta)
    return "stub"
# 還剩多少時間
def _time_left():
    if _deadline is None: return 0
    return max(0, _deadline - time.time())

# 定期存檔
def _maybe_periodic_full(opt):
    global _last_full_ts, _last_full_step
    now=time.time()
    due_time = (_PERIOD_SEC>0 and now - _last_full_ts >= _PERIOD_SEC)
    due_step = (_PERIOD_STEPS>0 and (_global_step - _last_full_step) >= _PERIOD_STEPS)
    if due_time or due_step:
        if torch.cuda.is_available(): torch.cuda.synchronize()
        _write_full(opt, "normal")
        _last_full_ts = now
        _last_full_step = _global_step


def _flush_two_tier_on_warn(opt):
    if _term_pending:
        return _write_stub("timeout_during_flush")
    if _pending_warn:
        if _time_left() >= _T_FULL + _SAFETY:
            if torch.cuda.is_available(): torch.cuda.synchronize()
            return _write_full(opt, "warn_flush")
        return _write_stub("timeout_during_flush")
    return "idle"

# 收到 SIGUSR1 訊號時, 計算 _deadline
def _on_warn(sig, frm):
    global _deadline, _pending_warn
    _deadline = time.time() + _WARN_SEC
    _pending_warn = True

# 收到 SIGTERM 訊號時, 表示即將被 kill
def _on_term(sig, frm):
    global _term_pending
    _term_pending = True

signal.signal(signal.SIGUSR1, _on_warn)
signal.signal(signal.SIGTERM, _on_term)

_orig_opt_step = optim.Optimizer.step


def _patched_opt_step(self, *args, **kwargs):
    global _global_step
    r = _orig_opt_step(self, *args, **kwargs)
    _global_step += 1
    _maybe_periodic_full(self)
    _flush_two_tier_on_warn(self)
    return r

optim.Optimizer.step = _patched_opt_step

def patch_all_optimizers():
    for name, cls in optim.__dict__.items():
        if isinstance(cls, type) and issubclass(cls, optim.Optimizer):
            if "step" in cls.__dict__:  # 子類別 override 了 step
                orig_step = cls.step
                def _patched(self, *args, __orig=orig_step, __cls=cls, **kwargs):
                    global _global_step
                    r = __orig(self, *args, **kwargs)
                    _global_step += 1
                    _maybe_periodic_full(self)
                    _flush_two_tier_on_warn(self)
                    return r
                cls.step = _patched

patch_all_optimizers()

# GradScaler 補丁, 存在 _last_seen_scaler
try:
    from torch.cuda.amp import GradScaler as _GS
    _orig_gs_step = _GS.step
    def _patched_gs_step(self, optimizer, *args, **kwargs):
        _last_seen_scaler[id(self)] = self
        return _orig_gs_step(self, optimizer, *args, **kwargs)
    _GS.step = _patched_gs_step
except Exception:
    pass

# LRScheduler 補丁, 存在_optimizer_to_scheduler
_orig_sched_step = None
try:
    _orig_sched_step = optim.lr_scheduler._LRScheduler.step
    def _patched_sched_step(self, *args, **kwargs):
        _optimizer_to_scheduler[self.optimizer] = self
        return _orig_sched_step(self, *args, **kwargs)
    optim.lr_scheduler._LRScheduler.step = _patched_sched_step
except Exception:
    pass

# ------- AUTO RESTORE ON OPTIMIZER CREATION -------
import builtins
_FAIR_AUTORESTORE = os.environ.get("FAIR_AUTORESTORE", "1") == "1"
_restored_once = False
_orig_opt_init = optim.Optimizer.__init__

def _auto_restore_try(opt):
    global _restored_once
    if _restored_once or not _FAIR_AUTORESTORE: return
    latest_path = os.path.join(_RUN_DIR, "LATEST")
    if not os.path.exists(latest_path): 
        _restored_once = True; return
    with open(latest_path, "r") as f:
        slot = f.read().strip()
    ckpt = os.path.join(_RUN_DIR, slot, "rank0.ckpt")
    if not os.path.exists(ckpt):
        _restored_once = True; return
    try:
        obj = torch.load(ckpt, map_location="cpu")
        model = _discover_model_from_optimizer(opt)
        if model is not None and obj.get("model") is not None:
            model.load_state_dict(obj["model"])
        opt.load_state_dict(obj["optimizer"])
        if obj.get("scaler") is not None:
            try:
                from torch.cuda.amp import GradScaler
                sc = GradScaler()
                sc.load_state_dict(obj["scaler"])
                _last_seen_scaler[id(sc)] = sc
            except Exception: pass
        sch = _get_scheduler_for_optimizer(opt)
        if sch is not None and obj.get("scheduler") is not None:
            sch.load_state_dict(obj["scheduler"])
        try:
            meta = obj.get("meta", {})
            globals()["_global_step"] = int(meta.get("global_step", 0))
            globals()["_last_success_ckpt_ts"] = int(meta.get("ts", 0))
        except Exception: pass
        print(f"[fair] auto-restore from {slot} OK, meta = {obj.get('meta', {})}")
    except Exception as e:
        print(f"[fair] auto-restore skipped: {e}")
    finally:
        _restored_once = True

# 猴補 Optimizer.__init__ → 建立 optimizer→model 對應 + 自動 restore
_orig_opt_init = optim.Optimizer.__init__

def _patched_opt_init(self, params, defaults):
    # 原本初始化
    _orig_opt_init(self, params, defaults)

    # ➊ 建立 optimizer→model 對應
    models = set()
    for p in params:
        if p in _param_to_model:
            models.add(_param_to_model[p])
    if models:
        _optimizer_to_model[self] = list(models) if len(models) > 1 else next(iter(models))
        print(f"[DEBUG] Optimizer {self.__class__.__name__} bound to models: {[m.__class__.__name__ for m in models]}", flush=True)

    _auto_restore_try(self)

optim.Optimizer.__init__ = _patched_opt_init
# ------- END AUTO RESTORE -------

