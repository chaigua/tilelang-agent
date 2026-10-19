import json, os, statistics, subprocess, sys, csv, glob
CORES=40; LOCK="/tmp/tilelang-msprof.lock"; OUT="op_cost_calibration.jsonl"
CONFIGS=[("vmax_chain",1,64,512,16),("vbrc_i",1,64,512,16),("vbrc_f",1,64,512,16),
         ("vcmp_gt",1,64,512,16),("vsel_gt",1,64,512,16),("vcmp_int",1,64,512,16),
         ("vsel_int",1,64,512,16),("vbrc_i",8,64,512,16),("vsel_gt",8,64,512,16)]
def measure(op,m,ch,reps,floor_reps):
    tag=f"p2_{op}_m{m}_ch{ch}_r{reps}"
    out=os.path.join("profiles","calibration","ops2",tag); os.makedirs(out,exist_ok=True); os.chmod(out,0o700)
    cmd=["flock",LOCK,"msprof","op","--kernel-name=main",f"--output={out}","--launch-count=20",
         "--warm-up=5","--dump=off","--aic-metrics=BasicInfo,PipeUtilization",
         sys.executable,"probe_ops2.py","--op",op,"--m",str(m),"--ch",str(ch),"--reps",str(reps),"--iters","25"]
    os.makedirs("logs",exist_ok=True)
    log=os.path.join("logs",f"opcost2_{tag}.log")
    with open(log,"w") as f:
        f.write(" ".join(cmd)+"\n"); f.flush()
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=3600); f.write(p.stdout+p.stderr)
    if p.returncode!=0: return {"tag":tag,"op":op,"m":m,"ch":ch,"reps":reps,"error":"msprof_rc"}
    durs=[];vec=[];sc=[]
    for path in glob.glob(os.path.join(out,"OPPROF_*","*","*","OpBasicInfo_*.csv")):
        for row in csv.DictReader(open(path)):
            if row.get("Task Duration(us)"): durs.append(float(row["Task Duration(us)"]))
    for path in glob.glob(os.path.join(out,"OPPROF_*","*","*","PipeUtilization_*.csv")):
        for row in csv.DictReader(open(path)):
            for key,col in (("v","aiv_vec_time(us)"),("s","aiv_scalar_time(us)")):
                v=row.get(col)
                if v not in (None,"","NA"): (vec if key=="v" else sc).append(float(v))
    if not durs: return {"tag":tag,"op":op,"m":m,"ch":ch,"reps":reps,"error":"no_rows"}
    return {"tag":tag,"op":op,"m":m,"ch":ch,"reps":reps,
            "duration_us":round(statistics.median(durs),3),
            "vec_us":round(statistics.median(vec),3) if vec else None,
            "scalar_us":round(statistics.median(sc),3) if sc else None,
            "lanes":m*ch,"msprof_raw_path":out}
for op,m,ch,reps,fr in CONFIGS:
    r=measure(op,m,ch,reps,fr); print(json.dumps(r),flush=True)
    open(OUT,"a").write(json.dumps(r)+"\n")
