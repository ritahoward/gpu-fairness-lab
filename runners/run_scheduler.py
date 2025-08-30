import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#!/usr/bin/env python3
import os, csv, time, json, signal, argparse, subprocess, pathlib, math
from datetime import datetime
from schedulers.timeslice_maxmin import TimesliceMaxMin
from schedulers.fifo_exclusive import FIFOExclusive

LOGS_DIR = "logs"

def load_jobs(csv_path):
    jobs = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            jobs.append({
                "job_id": r["job_id"],
                "user_id": r["user_id"],
                "submit_ts": int(r["submit_ts"]),    # 相對於實驗開始的秒數
                "script": r["script"],
                "args": r.get("args",""),
                "ckpt": r.get("ckpt","")             # 可選：固定 ckpt 路徑；若空則由 runner 統一下發
            })
    return sorted(jobs, key=lambda x: x["submit_ts"])

def ensure_dirs():
    pathlib.Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(LOGS_DIR, "ckpts")).mkdir(parents=True, exist_ok=True)

def now_ts():
    return time.time()

def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def launch_job(job, ckpt_path, extra_env=None):
    env = os.environ.copy()
    env["FAIR_CKPT_PATH"] = ckpt_path
    env["FAIR_JOB_ID"]    = job["job_id"]
    env["FAIR_USER_ID"]   = job["user_id"]
    if extra_env:
        env.update(extra_env)
    cmd = [job["script"]] + ([*job["args"].split()] if job["args"] else [])
    return subprocess.Popen(cmd, env=env)

def graceful_slice(p, slice_sec, warn_before=10, wait_timeout=60):
    t0 = now_ts()
    # 到片尾前一直睡
    while True:
        elapsed = now_ts() - t0
        if elapsed >= max(0, slice_sec - warn_before): break
        time.sleep(1)
    # 片尾預告 → 送 SIGUSR1
    try:
        os.kill(p.pid, signal.SIGUSR1)
    except ProcessLookupError:
        pass  # 可能已經自然結束
    # 給它時間 checkpoint 退出
    ckpt_overhead = None
    try:
        t_pre = now_ts()
        p.wait(timeout=wait_timeout)
        ckpt_overhead = now_ts() - t_pre
    except subprocess.TimeoutExpired:
        # 升級終止
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
        ckpt_overhead = wait_timeout + 10
    t1 = now_ts()
    used_wall = max(0, int(t1 - t0))
    return used_wall, ckpt_overhead, p.returncode if p.returncode is not None else -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--strategy", choices=["timeslice_maxmin","fifo_exclusive"], default="timeslice_maxmin")
    ap.add_argument("--slice-min", type=int, default=15)
    ap.add_argument("--obs-hours", type=float, default=4.0)
    ap.add_argument("--ckpt-root", default=os.path.join(LOGS_DIR, "ckpts"))
    args = ap.parse_args()

    ensure_dirs()
    jobs_all = load_jobs(args.workload)
    obs_start = now_ts()
    obs_end   = obs_start + args.obs_hours*3600

    # 初始化策略
    if args.strategy == "timeslice_maxmin":
        sched = TimesliceMaxMin(slice_sec=args.slice_min*60)
    else:
        sched = FIFOExclusive()

    # 狀態
    queue = []               # 已到達但尚未完成的 jobs
    finished = set()
    usage = {}               # user_id -> used_gpu_effective_sec（先用片長近似，之後可接 nvidia-smi 積分）
    slices = []              # 記每片

    current_job = None

    while now_ts() < obs_end:
        # 注入到達的工作
        sim_time = int(now_ts() - obs_start)
        while jobs_all and jobs_all[0]["submit_ts"] <= sim_time:
            queue.append(jobs_all.pop(0))

        # 移除已完成的
        queue = [j for j in queue if j["job_id"] not in finished]

        if not queue:
            time.sleep(1)
            continue

        pick = sched.pick_next(queue, usage)   # 必須回傳 job 物件
        if pick is None:
            time.sleep(1); continue

        # 決定 checkpoint 路徑（以 user/job 區分）
        ckpt_path = pick["ckpt"] or os.path.join(args.ckpt_root, f"{pick['user_id']}_{pick['job_id']}.pth")

        # 啟動
        p = launch_job(pick, ckpt_path)
        ts0 = now_ts()
        used_wall, ckpt_overhead, rc = graceful_slice(p, slice_sec=args.slice_min*60)

        # 這裡先用「片長」近似有效 GPU 時間；之後你可改為整合 nvidia-smi 的 sm_util 積分
        usage[pick["user_id"]] = usage.get(pick["user_id"], 0) + min(used_wall, args.slice_min*60)

        # 記錄切片
        slices.append({
            "ts_start": ts0,
            "ts_end": now_ts(),
            "user_id": pick["user_id"],
            "job_id": pick["job_id"],
            "pid": p.pid,
            "slice_sec": used_wall,
            "ckpt_overhead_sec": round(ckpt_overhead or 0, 2),
            "exit_code": rc
        })
        atomic_write_json(os.path.join(LOGS_DIR,"slices.json"), slices)
        atomic_write_json(os.path.join(LOGS_DIR,"user_usage.json"), usage)

        # 若 rc==10（自定義：表示作業自然完成），把它標成完成
        if rc == 10:
            finished.add(pick["job_id"])

    # 結束時存一次
    atomic_write_json(os.path.join(LOGS_DIR,"slices.json"), slices)
    atomic_write_json(os.path.join(LOGS_DIR,"user_usage.json"), usage)
    print("Done. Logs saved to logs/")

if __name__ == "__main__":
    main()

