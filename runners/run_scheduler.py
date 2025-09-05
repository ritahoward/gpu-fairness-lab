'''

根據 CSV 定義的工作（job）到達時間與執行腳本，使用你選定的排程策略（Max-Min 或 FIFO 獨佔），在「片尾前 warn 幾秒送 SIGUSR1」的機制下，等待作業自行 checkpoint 並退出；若沒在時限內退出，則升級為 terminate → kill。每一片的使用情況與 checkpoint 開銷都寫進 logs/ 的 JSON 檔，usage 用「片長近似有效 GPU 時間」。
近似的意思：
如果某個人真的跑滿 15 分鐘，那就算他用了 15 分鐘的 GPU。
如果他只跑 10 分鐘就自己結束，那就算他用了 10 分鐘的 GPU。
如果他跑到 15 分鐘但 GPU 其實沒滿載（例如只用到一半效能），程式還是當成他用了 15 分鐘的 GPU。（也就是說，程式只是「看時間」來估算，假設 GPU 在這段時間裡都是滿載運算，沒有去真的量 GPU 使用率。）

'''
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#!/usr/bin/env python3
import os, csv, time, json, signal, argparse, subprocess, pathlib, math
from datetime import datetime, timezone
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
    #return time.time()
    return time.monotonic()
    
def wall_now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=2)
    os.replace(tmp, path)
    
'''

def launch_job(job, ckpt_path, extra_env=None):
    env = os.environ.copy()
    env["FAIR_CKPT_PATH"] = ckpt_path
    env["FAIR_JOB_ID"]    = job["job_id"]
    env["FAIR_USER_ID"]   = job["user_id"]
    if extra_env:
        env.update(extra_env)
    cmd = [job["script"]] + ([*job["args"].split()] if job["args"] else [])
    return subprocess.Popen(cmd, env=env)
    
    

'''

def launch_job(job, ckpt_path, extra_env=None):
    env = os.environ.copy()
    env["FAIR_CKPT_PATH"] = ckpt_path
    env["FAIR_JOB_ID"]    = job["job_id"]
    env["FAIR_USER_ID"]   = job["user_id"]
    if extra_env:
        env.update(extra_env)

    payload = [job["script"]] + ([*job["args"].split()] if job["args"] else [])
    payload_str = " ".join(map(str, payload))

    # 若 runner 本身是被 sbatch 起來的，可以用 srun 開 step（帳務/cgroup 友善）
    if os.getenv("SLURM_JOB_ID"):
        cmd = ["srun", "--exclusive", "--gres=gpu:1", "--mpi=none", "bash", "-lc", payload_str]
    else:
        cmd = payload

    return subprocess.Popen(cmd, env=env)



'''

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
''' 

'''   
def graceful_slice(p, slice_sec, warn_before=5, wait_timeout=60):
    t0 = now_ts()
    while True:
        elapsed = now_ts() - t0
        if elapsed >= max(0, slice_sec - warn_before): break
        time.sleep(1)
    try:
        os.kill(p.pid, signal.SIGUSR1)
    except ProcessLookupError:
        pass
    ckpt_overhead = None
    try:
        t_pre = now_ts()
        p.wait(timeout=wait_timeout)
        ckpt_overhead = now_ts() - t_pre
    except subprocess.TimeoutExpired:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
        ckpt_overhead = wait_timeout + 10
    t1 = now_ts()
    used_wall = max(0, int(t1 - t0))
    return used_wall, ckpt_overhead, p.returncode if p.returncode is not None else -1    
    
'''
    
    
def graceful_slice(p, slice_sec, warn_before=5, wait_timeout=60):
    t0 = now_ts()
    deadline_warn = t0 + max(0, slice_sec - warn_before)

    # 1) 片尾前監看，若子行程早退就提前結束
    exited_early = False
    while True:
        if p.poll() is not None:
            exited_early = True
            break
        now = now_ts()
        if now >= deadline_warn:
            break
        time.sleep(min(0.1, max(0, deadline_warn - now)))

    # 2) 若還活著 → 發預警
    signaled = False
    if p.poll() is None:
        try:
            os.kill(p.pid, signal.SIGUSR1)
            signaled = True
        except ProcessLookupError:
            pass

    # 3) 等 checkpoint 退出或升級終止
    ckpt_overhead = 0.0
    exit_reason = None
    if p.poll() is not None:
        # 已在 warn 之前自然結束
        exit_reason = "natural_early"
    else:
        t_pre = now_ts()
        try:
            p.wait(timeout=wait_timeout)
            ckpt_overhead = now_ts() - t_pre
            exit_reason = "ckpt_exit" if signaled else "natural"
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                p.wait(timeout=10)
                ckpt_overhead = now_ts() - t_pre
                exit_reason = "terminated"
            except subprocess.TimeoutExpired:
                p.kill()
                ckpt_overhead = now_ts() - t_pre
                exit_reason = "killed"

    t1 = now_ts()
    used_wall = max(0.0, t1 - t0)
    rc = p.returncode if p.returncode is not None else -1
    return used_wall, ckpt_overhead, rc, exit_reason


    

    
    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--strategy", choices=["timeslice_maxmin","fifo_exclusive"], default="timeslice_maxmin")
    ap.add_argument("--slice-min", type=int, default=15)
    ap.add_argument("--obs-hours", type=float, default=4.0)
    ap.add_argument("--ckpt-root", default=os.path.join(LOGS_DIR, "ckpts"))
    ap.add_argument("--warn-sec", type=int, default=5, help="片尾提前幾秒發 SIGUSR1")
    ap.add_argument("--exit-timeout-sec", type=int, default=60, help="等 checkpoint 退出的上限")

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
        # used_wall, ckpt_overhead, rc = graceful_slice(p, slice_sec=args.slice_min*60)
        used_wall, ckpt_overhead, rc, exit_reason = graceful_slice(
    	    p,
    	    slice_sec=args.slice_min*60,
    	    warn_before=args.warn_sec,
    	    wait_timeout=args.exit_timeout_sec
	)


        # 這裡先用「片長」近似有效 GPU 時間；之後你可改為整合 nvidia-smi 的 sm_util 積分
        usage[pick["user_id"]] = usage.get(pick["user_id"], 0) + min(used_wall, args.slice_min*60)

        # 記錄切片
        slices.append({
            "ts_start_mono": ts0,
            "ts_end_mono": now_ts(),
            "ts_start_wall": wall_now_iso(),
            "ts_end_wall": wall_now_iso(),
            "user_id": pick["user_id"],
            "job_id": pick["job_id"],
            "pid": p.pid,
            "slice_sec": round(used_wall, 3),
            "ckpt_overhead_sec": round(ckpt_overhead or 0, 3),
            "exit_code": rc,
            "exit_reason": exit_reason
        })

        '''
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
        '''
        
        atomic_write_json(os.path.join(LOGS_DIR,"slices.json"), slices)
        atomic_write_json(os.path.join(LOGS_DIR,"user_usage.json"), usage)

            
        if exit_reason in ("natural_early", "natural", "ckpt_exit") and rc in (0, 10):
            finished.add(pick["job_id"])


    # 結束時存一次
    atomic_write_json(os.path.join(LOGS_DIR,"slices.json"), slices)
    atomic_write_json(os.path.join(LOGS_DIR,"user_usage.json"), usage)
    print("Done. Logs saved to logs/")

if __name__ == "__main__":
    main()

