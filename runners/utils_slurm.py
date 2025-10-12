import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#!/usr/bin/env python3
import subprocess, time, datetime


def _run(cmd):
    #print(f"[run] {' '.join(cmd)}")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0: raise RuntimeError(p.stderr.strip())
    return p.stdout



def list_fair_jobs():
    fmt="%i|%u|%T|%V|%j"
    out=_run(["squeue","-h","-o",fmt])
    jobs=[]
    for line in out.strip().splitlines():
        jobid,user,state,subm,name=line.split("|",4)
        if not name.startswith("FAIR_"): continue
        ts=int(datetime.datetime.fromisoformat(subm.replace("Z","")).timestamp()) if subm!="N/A" else int(time.time())
        jobs.append({"jobid":jobid,"user":user,"state":state,"submit_ts":ts,"name":name})
    print("[debug:list_fair_jobs] collected jobs:")
    for j in jobs:
        print(f"  jobid={j['jobid']}, user={j['user']}, state={j['state']}, name={j['name']}")

    return jobs

def hold(jobid): _run(["scontrol","hold",str(jobid)])
def release(jobid): _run(["scontrol","release",str(jobid)])
def signal(jobid,sig): _run(["scancel",f"--signal={sig}",str(jobid)])
'''
def requeuehold(jobid):
    try: 
    	_run(["scontrol","requeue",str(jobid)])
    	_run(["bash", "-c", f"scontrol update JobId={jobid} StartTime=now"])
    	#_run(["scontrol", "update", f"JobId={jobid}", "StartTime=now"])
    	print(f"[server] Job {jobid} requeued StartTime now")
	#print(f"[server] Job {jobid} requeued StartTime now")
    except: 
    	_run(["scontrol","requeue",str(jobid)])
    	_run(["scontrol","hold",str(jobid)])
'''
    	
def requeuehold(jobid):
    try:
        _run(["scontrol", "requeue", str(jobid)])
        #_run(["scontrol update JobId={jobid} StartTime=now"])
        time.sleep(1.0)
        _run(["scontrol", "release", str(jobid)])
        _run(["scontrol", "update", f"JobId={jobid}", "StartTime=now"])
        #_run(["scontrol", "show", "job", str(jobid)])
        print(f"[server] Job {jobid} requeued + StartTime=now + released")
    except subprocess.CalledProcessError as e:
        print(f"[server] requeuehold failed at {e.cmd} → returncode={e.returncode}")
    except Exception as e:
        print(f"[server] Warning: failed to requeuehold {jobid}: {e}")


'''
def state(jobid):
    out=_run(["squeue","-h","-j",str(jobid),"-o","%T"])
    return out.strip() if out.strip()!="" else "GONE"
'''
    
def state(jobid):
    out = _run(["squeue","-h","-j",str(jobid),"-o","%t"])
    return out.strip() if out.strip()!="" else "GONE"


def wait_for_state(jobid, targets, timeout_sec, poll=1.0):
    t0=time.time()
    while time.time()-t0<timeout_sec:
        st=state(jobid)
        if st in targets: return st
        time.sleep(poll)
    return state(jobid)
    
def ensure_held_at_boot(jobs, tracked):
    """確保新偵測到的 job 都先被 hold"""
    for j in jobs:
        jid = j["jobid"]
        if jid not in tracked:
            print(f"[server] New job detected: {jid}, sending hold...")
            try:
                _run(["scontrol", "hold", str(jid)])
                print(f"[server] Job {jid} held successfully.")
            except Exception as e:
                print(f"[server] Warning: failed to hold job {jid}: {e}")
            tracked.add(jid)


