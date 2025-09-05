import os,sys,time,signal,torch
should_ckpt=False
def on_usr1(sig,frame):
    global should_ckpt
    should_ckpt=True
    print("[SIGUSR1] received",flush=True)
signal.signal(signal.SIGUSR1,on_usr1)
ckpt=os.getenv("FAIR_CKPT_PATH","/tmp/min.ckpt")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
it=0
def gpu_step():
    x=torch.randn(2048,2048,device=device)
    for _ in range(4): x=x@x
while True:
    it+=1
    gpu_step()
    if it%10==0: print(f"[RUN_train_ckpt] it={it}",flush=True)
    if should_ckpt:
        t0=time.time()
        torch.save({"it":it},ckpt)
        dt=time.time()-t0
        print(f"[CKPT] saved it={it} t={dt:.2f}s path={ckpt}",flush=True)
        sys.exit(0)

