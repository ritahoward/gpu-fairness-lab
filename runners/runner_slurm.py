import subprocess, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#!/usr/bin/env python3

import argparse, time, json, os, importlib, socket
from runners.utils_slurm import list_fair_jobs, hold, release, signal, requeuehold, state, wait_for_state, ensure_held_at_boot
#from runners.runner_signal import send_usr1

usage = {}
last_submit = {}   # ✅ 全域字典：用來記錄 requeue 後的模擬提交時間



def load_scheduler(name):
    mod=importlib.import_module(f"schedulers.{name}")
    cls=[getattr(mod,x) for x in dir(mod) if not x.startswith("_") and isinstance(getattr(mod,x),type)][0]
    return cls()
    
def get_job_node(jobid):
    out = subprocess.check_output(["squeue", "-h", "-j", str(jobid), "-o", "%N"], text=True)
    return out.strip()

'''
def send_usr1(jobid):
    node = subprocess.check_output(
        ["squeue", "-h", "-j", str(jobid), "-o", "%N"], text=True
    ).strip()
    local = socket.gethostname()
    cmd = f"pgrep -f FAIR_JOB_ID={jobid} | xargs -r kill -USR1"

    if node == local:
        print(f"[server] sending SIGUSR1 to job {jobid} locally")
        os.system(cmd)
    else:
        print(f"[server] sending SIGUSR1 to job {jobid} via SSH ({node})")
        os.system(f"ssh -o StrictHostKeyChecking=no {node} '{cmd}'")
'''


def send_usr1(jobid):
    try:
        node = subprocess.check_output(
            ["squeue", "-h", "-j", str(jobid), "-o", "%N"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        print(f"[server] send_usr1: cannot find node for job {jobid}")
        return
    local = socket.gethostname()
    cmd = f"pgrep -f FAIR_JOB_ID={jobid} | xargs -r kill -USR1"
    if node == local or node == "localhost":
        print(f"[server] sending SIGUSR1 to job {jobid} locally")
        os.system(cmd)
    else:
        print(f"[server] sending SIGUSR1 to job {jobid} via SSH ({node})")
        os.system(f"ssh -o StrictHostKeyChecking=no {node} '{cmd}'")



  


'''

def choose(queue, usage, scheduler):
    # 允許 Held 狀態參與排程
    q = [dict(x) for x in queue if x["state"] in ("PD","R","CF","CG","S","RD","RF","RH","RQ","JobHeldUser","JOB_HELD")]
    if not q:
        print("[debug] choose(): no eligible jobs after filtering")
        return None

    for x in q:
        x["user_id"] = x["user"]

    # 若 scheduler 有 pick_next 方法就用它，否則用提交時間排序
    return scheduler.pick_next(q, usage) if hasattr(scheduler, "pick_next") else sorted(q, key=lambda x: x["submit_ts"])[0]
'''

'''
def choose(queue, usage, scheduler):
    # 統一轉大寫，並允許 Held / Pending / Running 狀態
    global last_submit
    q = []
    for x in queue:
        st = x["state"].upper()
        if st in ("PD", "PENDING", "R", "RUNNING", "CF", "CG", "S", "RD", "RF", "RH", "RQ", "HELD", "JOBHELDUSER", "JOB_HELD"):
            q.append(dict(x))

    if not q:
        print("[debug] choose(): no eligible jobs after filtering")
        return None

    for x in q:
        x["user_id"] = x["user"]

    # 呼叫 scheduler 選擇下一個 job
    #return scheduler.pick_next(q, usage) if hasattr(scheduler, "pick_next") else sorted(q, key=lambda x: x["submit_ts"])[0]
    return sorted(q, key=lambda x: last_submit.get(x["jobid"], x["submit_ts"]))[0]
'''

'''
def choose(queue, usage, scheduler):
    global last_submit
    q = []
    for x in queue:
        st = x["state"].upper()
        if st in ("PD", "PENDING", "R", "RUNNING", "CF", "CG", "S", "RD", "RF", "RH", "RQ", "HELD", "JOBHELDUSER", "JOB_HELD"):
            q.append(dict(x))

    if not q:
        print("[debug] choose(): no eligible jobs after filtering")
        return None

    for x in q:
        x["user_id"] = x["user"]

    chosen = sorted(q, key=lambda x: last_submit.get(x["jobid"], x["submit_ts"]))[0]
    print(f"[debug] choose() ordering: {[ (x['jobid'], last_submit.get(x['jobid'], x['submit_ts'])) for x in q ]}")
    return chosen
'''
    
def choose(q, usage, scheduler, last_submit):
    """
    從目前 queue (q) 中選擇下一個要執行的 job。
    若 scheduler 實作 pick_next(queue, last_submit)，則呼叫該方法；
    否則預設以 last_submit 或 submit_ts 做排序。
    """
    if not q:
        return None

    # Debug: 檢查每個 job 的排序鍵
    print("[debug] choose() ordering:", [(x["jobid"], last_submit.get(x["jobid"], x["submit_ts"])) for x in q])

    # 呼叫 scheduler.pick_next() 若有實作
    if hasattr(scheduler, "pick_next"):
        chosen = scheduler.pick_next(q, last_submit)
    else:
        # fallback: 用 last_submit 排序
        chosen = sorted(q, key=lambda x: last_submit.get(x["jobid"], x["submit_ts"]))[0]

    print(f"[debug] choose() selected job={chosen['jobid']}")
    return chosen




'''
        
def ensure_held_at_boot(jobs, tracked):
    """確保新偵測到的 job 都先被 hold"""
    for j in jobs:
        jid = j["jobid"]
        if jid in tracked:
            continue
        print(f"[server] New job detected: {jid}, sending hold...")
        try:
            hold(jid)
            print(f"[server] Job {jid} held successfully.")
        except Exception as e:
            print(f"[server] Warning: failed to hold job {jid}: {e}")
        tracked.add(jid)
'''

def ensure_held_at_boot(jobs, tracked):
    """確保所有 FAIR_* job 一律進入 hold 狀態"""
    for j in jobs:
        jid = j["jobid"]
        st = j["state"].upper()
        if jid in tracked:
            continue

        # 若 job 尚未 hold，立即嘗試 hold
        if st not in ("JOBHELDUSER", "JOB_HELD", "HELD"):
            print(f"[server] Holding job {jid} (state={st})...")
            try:
                hold(jid)
                print(f"[server] Job {jid} is now held.")
            except Exception as e:
                print(f"[server] Warning: failed to hold job {jid}: {e}")

        tracked.add(jid)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--slice-sec",type=int,default=300)
    ap.add_argument("--warn-sec",type=int,default=60)
    ap.add_argument("--grace-sec",type=int,default=60)
    ap.add_argument("--poll-sec",type=float,default=1.0)
    ap.add_argument("--scheduler",type=str,default="fifo_exclusive")
    ap.add_argument("--logdir",type=str,default="logs")
    args=ap.parse_args()

    os.makedirs(args.logdir,exist_ok=True)
    usage={}
    last_submit = {}

    tracked=set()
    sched=load_scheduler(args.scheduler)
    cur=None
    
    while True:
        jobs = list_fair_jobs()
        if not jobs:
            print("[server] No FAIR_* jobs found, sleeping...")
            time.sleep(args.poll_sec)
            continue

        print(f"[server] Detected jobs: {[j['jobid'] for j in jobs]}")
        ensure_held_at_boot(jobs, tracked)
        nxt = choose(jobs, usage, sched, last_submit)

        
        #######################
        print(f"[debug] queue size={len(jobs)} sched={sched}")
        print(f"[debug] scheduler returned: {nxt}")
        #######################

        
        

        if nxt is None:
            print("[server] No job selected by scheduler, sleeping...")
            time.sleep(args.poll_sec)
            continue
            



        jid = nxt["jobid"]
        user = nxt["user"]
        print(f"[server] ===> Selecting job {jid} (user={user}) for next slice")

        # 若目前有正在跑的任務但要換人
        '''
        if cur and cur != jid:
            print(f"[server] Sending SIGUSR1 to current job {cur}")
            try:
                signal(cur, "USR1")
            except Exception as e:
                print(f"[server] Warning: failed to signal {cur}: {e}")
            st = wait_for_state(cur, ("COMPLETED", "CANCELLED", "GONE", "FAILED", "TIMEOUT"), args.grace_sec)
            print(f"[server] Current job {cur} exited with state {st}")
            try:
                requeuehold(cur)
            except Exception as e:
                print(f"[server] Warning: failed to requeuehold {cur}: {e}")
            cur = None
        '''
        
        if cur:
            print(f"[server] Sending SIGUSR1 to current job {cur}")
            try:
                #send_usr1(cur)
                print(f"[server] Sending SIGUSR1 to current job {cur}")
                try:
                    print(f"[debug] About to send_usr1({cur})")
                    send_usr1(cur)
                    print(f"[debug] Finished send_usr1({cur})")
                except Exception as e:
                    print(f"[server] Warning: failed to send_usr1 {cur}: {e}")


            except Exception as e:
                print(f"[server] Warning: failed to signal {cur}: {e}")

            st = wait_for_state(cur, ("COMPLETED", "CANCELLED", "GONE", "FAILED", "TIMEOUT"), args.grace_sec)
            print(f"[server] Current job {cur} exited with state {st}")
            

            #try:
            #    requeuehold(cur)
            try:
                print(f"[debug] calling requeuehold({cur})")
                requeuehold(cur)
                print(f"[debug] requeuehold({cur}) finished OK")
                #st = wait_for_state(cur, ("PD", "PENDING"), 5)
                last_submit[cur] = time.time()  # 模擬重新提交時間
                print(f"[server] Job {cur} requeued and held (ts updated at {last_submit[cur]:.0f})")
                #print(f"[server] Job {cur} requeued and held for next round (ts updated)")
            except Exception as e:
                print(f"[server] Warning: failed to requeuehold {cur}: {e}")

                
            
            cur = None



        # 放行下一個
        try:
            release(jid)
            print(f"[server] Released job {jid}, waiting for it to start running...")
        except Exception as e:
            print(f"[server] Warning: failed to release {jid}: {e}")

        st = wait_for_state(jid, ("R",), args.slice_sec)
        if st != "R":
            print(f"[server] Job {jid} not running (state={st}), skipping...")
            try:
                hold(jid)
            except:
                pass
            time.sleep(args.poll_sec)
            continue

        cur = jid
        t0 = time.time()
        print(f"[server] Job {jid} started running, slice={args.slice_sec}s warn={args.warn_sec}s")
        print(f"[debug] entering slice loop for {jid}")

        # 等待 slice
        while time.time() - t0 < args.slice_sec - args.warn_sec:
            time.sleep(args.poll_sec)
            if state(jid) != "R":
                break

        print(f"[server] Slice ending, sending SIGUSR1 to {jid}")
        try:
            #signal(jid, "USR1")
            print(f"[server] Sending SIGUSR1 to current job {jid}")
            try:
                print(f"[debug] About to send_usr1({jid})")
                send_usr1(jid)
                print(f"[debug] Finished send_usr1({jid})")
            except Exception as e:
                print(f"[server] Warning: failed to send_usr1 {jid}: {e}")

            
        except Exception as e:
            print(f"[server] Warning: failed to send SIGUSR1: {e}")
        st = wait_for_state(jid, ("COMPLETED", "CANCELLED", "GONE", "FAILED", "TIMEOUT"), args.grace_sec)
        print(f"[server] Job {jid} ended slice with state {st}")

        try:
            print(f"[debug] calling requeuehold({jid})")
            requeuehold(jid)
            print(f"[debug] requeuehold({jid}) finished OK")
            #st = wait_for_state(jid, ("PD", "PENDING"), 5)
            last_submit[jid] = time.time()  # 模擬重新提交時間
            print(f"[server] Job {jid} requeued and held (ts updated at {last_submit[cur]:.0f})")
            #print(f"[server] Job {jid} requeued and held for next round")
        except Exception as e:
            print(f"[server] Warning: failed to requeuehold {jid}: {e}")
        
        

        cur = None
        usage[user] = usage.get(user, 0) + args.slice_sec
        with open(os.path.join(args.logdir, "usage.jsonl"), "a") as f:
            f.write(json.dumps({"ts": time.time(), "user": user, "add": args.slice_sec, "sum": usage[user], "jobid": jid}) + "\n")


if __name__=="__main__":
    main()

