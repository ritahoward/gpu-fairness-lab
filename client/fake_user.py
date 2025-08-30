# client/fake_user.py
import os, signal, sys, time

should_exit = False
def on_sigusr1(sig, frame):
    global should_exit
    should_exit = True

signal.signal(signal.SIGUSR1, on_sigusr1)

uid = os.getenv("FAIR_USER_ID", "uX")
jid = os.getenv("FAIR_JOB_ID", "JX")

print(f"[{uid}/{jid}] start pid={os.getpid()}", flush=True)

t0 = time.time()
while True:
    time.sleep(1)                  # 假裝在算
    if should_exit:
        # 模擬 checkpoint 完成
        print(f"[{uid}/{jid}] got SIGUSR1, exiting...", flush=True)
        sys.exit(0)                # 0 代表「正常換片退出」

