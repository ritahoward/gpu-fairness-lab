import sys,time,torch,signal
def on_usr1(sig,frame):
    print("[SIGUSR1] ignored",flush=True)
signal.signal(signal.SIGUSR1,on_usr1)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
it=0
def gpu_step():
    x=torch.randn(2048,2048,device=device)
    for _ in range(4): x=x@x
while True:
    it+=1
    gpu_step()
    if it%10==0: print(f"[RUN_train_ignore] it={it}",flush=True)
    time.sleep(0.05)

