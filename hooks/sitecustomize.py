import os, time, json, signal, random, shutil, gc, weakref, types
import torch
import torch.nn as nn
import torch.optim as optim

_RUN_DIR = os.environ.get("FAIR_RUN_DIR", "./runs/exp1")
_PERIOD_SEC = int(os.environ.get("FAIR_CKPT_PERIOD_SEC", "600"))
_PERIOD_STEPS = int(os.environ.get("FAIR_CKPT_PERIOD_STEPS", "0"))
_T_FULL = int(os.environ.get("T_FULL_SEC", "40"))
_SAFETY = int(os.environ.get("CKPT_SAFETY_SEC", "5"))
_WARN_SEC = int(os.environ.get("WARN_SEC", "120"))

os.makedirs(_RUN_DIR, exist_ok=True)

class _ABWriter:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.slot_names = ["ckpt_slotA","ckpt_slotB"]
        for s in self.slot_names:
            os.makedirs(os.path.join(run_dir, s), exist_ok=True)
        self.tmp = os.path.join(run_dir, "tmp"); os.makedirs(self.tmp, exist_ok=True)
    def _active(self):
        p = os.path.join(self.run_dir, "LATEST")
        if not os.path.exists(p): return self.slot_names[0]
        with open(p,"r") as f: cur=f.read().strip()
        return cur if cur in self.slot_names else self.slot_names[0]
    def _next(self):
        return self.slot_names[1] if self._active()==self.slot_names[0] else self.slot_names[0]
    def _atomic_bytes(self, path, data:bytes):
        d=os.path.dirname(path); os.makedirs(d, exist_ok=True)
        tmp=os.path.join(self.tmp, f".tmp_{int(time.time()*1e6)}_{random.randint(0,1<<16)}")
        with open(tmp,"wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    def _atomic_json(self, path, obj):
        self._atomic_bytes(path, json.dumps(obj, ensure_ascii=False).encode("utf-8"))
    def _atomic_copy(self, src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp=os.path.join(self.tmp, f".tmp_{int(time.time()*1e6)}_{random.randint(0,1<<16)}")
        with open(src,"rb") as s, open(tmp,"wb") as t:
            shutil.copyfileobj(s,t, length=1<<20); t.flush(); os.fsync(t.fileno())
        os.replace(tmp,dst)
    def write_latest(self, slot):
        latest=os.path.join(self.run_dir,"LATEST"); tmp=latest+".tmp"
        with open(tmp,"w") as f:
            f.write(slot); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, latest)
    def save_state(self, slot, fname, obj):
        tmpf=os.path.join(self.tmp, f".state_{int(time.time()*1e6)}_{random.randint(0,1<<16)}.pt")
        torch.save(obj, tmpf)
        self._atomic_copy(tmpf, os.path.join(self.run_dir, slot, fname))
        try: os.remove(tmpf)
        except: pass
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
_optimizer_to_scheduler = weakref.WeakKeyDictionary()


def _ddp_is_init():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def _rank():
    return torch.distributed.get_rank() if _ddp_is_init() else 0

def _barrier():
    if _ddp_is_init(): torch.distributed.barrier()


def _discover_model_from_optimizer(opt):
    params = set([p for g in opt.param_groups for p in g.get('params',[]) if isinstance(p, torch.Tensor)])
    best=None; best_cnt=-1
    for obj in gc.get_objects():
        try:
            if isinstance(obj, nn.Module):
                ps=set([p for p in obj.parameters(recurse=True)])
                c=len(params & ps)
                if c>best_cnt:
                    best=obj; best_cnt=c
        except: pass
    return best


def _get_scheduler_for_optimizer(opt):
    sch=_optimizer_to_scheduler.get(opt)
    if sch is not None: return sch
    for obj in gc.get_objects():
        try:
            if isinstance(obj, optim.lr_scheduler._LRScheduler) and getattr(obj, 'optimizer', None) is opt:
                _optimizer_to_scheduler[opt]=obj
                return obj
        except: pass
    return None


def _pack_full(opt, reason="normal"):
    model = _discover_model_from_optimizer(opt)
    scaler = None
    for k,v in list(_last_seen_scaler.items()): scaler=v
    scheduler = _get_scheduler_for_optimizer(opt)
    dl_state = None
    ema_state = None
    meta = {"ts": int(time.time()), "level":"full", "trigger": reason, "reason":"none", "global_step": int(_global_step), "lr_step": int(getattr(scheduler,'last_epoch',0)), "rank": _rank(), "last_success_ckpt_ts": int(_last_success_ckpt_ts)}
    return {"model": (model.state_dict() if model is not None else None),
            "optimizer": opt.state_dict(),
            "scaler": (scaler.state_dict() if scaler is not None else None),
            "scheduler": (scheduler.state_dict() if scheduler is not None else None),
            "dataloader": dl_state,
            "ema": ema_state,
            "meta": meta}, meta


def _write_full(opt, reason):
    global _last_success_ckpt_ts
    if _rank()!=0:
        _barrier(); return "skipped_nonzero"
    slot=_writer._next()
    state, meta = _pack_full(opt, reason)
    _writer.save_state(slot, "rank0.ckpt", state)
    _writer.save_json(slot, "meta.json", state["meta"])
    _writer.finalize(slot)
    _last_success_ckpt_ts = int(time.time())
    _barrier()
    return "full"


def _write_stub(reason="timeout_during_flush"):
    if _rank()!=0:
        _barrier(); return "stub"
    slot=_writer._next()
    meta={"ts": int(time.time()), "level":"stub", "trigger":"warn_flush", "reason": reason, "global_step": int(_global_step), "lr_step": 0, "rank": 0, "last_success_ckpt_ts": int(_last_success_ckpt_ts)}
    _writer.save_json(slot, "meta.json", meta)
    _barrier()
    return "stub"


def _time_left():
    if _deadline is None: return 0
    return max(0, _deadline - time.time())


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


def _on_warn(sig, frm):
    global _deadline, _pending_warn
    _deadline = time.time() + _WARN_SEC
    _pending_warn = True

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

try:
    from torch.cuda.amp import GradScaler as _GS
    _orig_gs_step = _GS.step
    def _patched_gs_step(self, optimizer, *args, **kwargs):
        _last_seen_scaler[id(self)] = self
        return _orig_gs_step(self, optimizer, *args, **kwargs)
    _GS.step = _patched_gs_step
except Exception:
    pass

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
        print(f"[fair] auto-restore from {slot} OK")
    except Exception as e:
        print(f"[fair] auto-restore skipped: {e}")
    finally:
        _restored_once = True

def _patched_opt_init(self, params, defaults):
    _orig_opt_init(self, params, defaults)
    _auto_restore_try(self)

optim.Optimizer.__init__ = _patched_opt_init
# ------- END AUTO RESTORE -------

