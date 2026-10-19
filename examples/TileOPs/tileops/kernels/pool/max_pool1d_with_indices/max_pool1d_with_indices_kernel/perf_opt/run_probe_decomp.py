import sys, os, subprocess, csv, glob, statistics, json
LOCK="/tmp/tilelang-msprof.lock"
variants = sys.argv[1:]
cases = ["sincnet-speaker-local-float16"]
for v in variants:
    impl = f"_probe_{v}.py"
    for case in cases:
        out = os.path.join("profiles","probe","decomp",v,case)
        os.makedirs(out, exist_ok=True); os.chmod(out,0o700)
        cmd=["flock",LOCK,"msprof","op","--kernel-name=main",f"--output={out}",
             "--launch-count=20","--warm-up=5","--dump=off",
             "--aic-metrics=BasicInfo,PipeUtilization,Memory",
             "/workspace/venv/bin/python3","bench.py","--impl",impl,"--case",case,"--iters","25"]
        log=os.path.join("logs",f"probe_decomp_{v}_{case}.log")
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=3600)
        open(log,"w").write(" ".join(cmd)+"\n"+p.stdout+p.stderr)
        if p.returncode!=0:
            print(f"{v:24s} {case}: FAILED rc={p.returncode}"); continue
        durs=[];vec=[];sc=[];mte2=[];mte3=[]
        for path in glob.glob(os.path.join(out,"OPPROF_*","*","*","OpBasicInfo_*.csv")):
            for row in csv.DictReader(open(path)):
                if row.get("Task Duration(us)"): durs.append(float(row["Task Duration(us)"]))
        for path in glob.glob(os.path.join(out,"OPPROF_*","*","*","PipeUtilization_*.csv")):
            for row in csv.DictReader(open(path)):
                for k,c in (("v","aiv_vec_time(us)"),("s","aiv_scalar_time(us)"),("2","aiv_mte2_time(us)"),("3","aiv_mte3_time(us)")):
                    x=row.get(c)
                    if x not in (None,"","NA"):
                        {"v":vec,"s":sc,"2":mte2,"3":mte3}[k].append(float(x))
        m=lambda a: round(statistics.median(a),2) if a else None
        print(f"{v:24s} {case}: T={m(durs)} vec={m(vec)} scalar={m(sc)} mte2={m(mte2)} mte3={m(mte3)}", flush=True)
