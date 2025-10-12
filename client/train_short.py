import time,torch,signal
def on_usr1(sig,frame):
    print("[SIGUSR1] received",flush=True)
signal.signal(signal.SIGUSR1,on_usr1)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
for it in range(1,21):
    x=torch.randn(2048,2048,device=device)
    x=x@x
    if it%5==0: print(f"[RUN_train_short] it={it}",flush=True)
    time.sleep(0.05)
print("[DONE] natural early exit",flush=True)

