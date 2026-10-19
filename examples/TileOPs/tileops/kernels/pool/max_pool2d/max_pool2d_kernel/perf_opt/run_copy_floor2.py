"""Runner for the DMA-ceiling probes (copy_floor2.py) -> copy_floor_calibration.jsonl."""
import csv, datetime, glob, json, os, statistics, subprocess, sys
from copy_floor2 import SHAPES, VARIANTS

OUT = "copy_floor_calibration.jsonl"
LAUNCH_COUNT = 20
WARM_UP = 5
METRICS = "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB"
TS = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def spans(v):
    shape, ch, rr, wr, n_units, kind = VARIANTS[v]
    n, c, h, w, out_w, s_h, k_h = SHAPES[shape]
    rs, ws = rr * w, wr * out_w
    if kind == "write":
        rs = ws
    return shape, ch, rr, wr, n_units, kind, rs, ws

def expected(v):
    shape, ch, rr, wr, n_units, kind, rs, ws = spans(v)
    n, c, h, w, out_w, _, _ = SHAPES[shape]
    num_cblk = (c + ch - 1) // ch
    units = n * n_units * num_cblk
    rd = 0 if kind == "write" else units * rs * ch * 2
    wr = units * ws * ch * 2
    return rd, wr, units

def parse(d):
    durs, blocks = [], None
    for p in sorted(glob.glob(os.path.join(d,"OPPROF_*","*","*","OpBasicInfo_*.csv"))):
        for row in csv.DictReader(open(p)):
            if row.get("Task Duration(us)"):
                durs.append(float(row["Task Duration(us)"])); blocks = row.get("Block Dim", blocks)
    eng={k:[] for k in ("aiv_time","vec","mte2","mte3","scalar")}
    for p in glob.glob(os.path.join(d,"OPPROF_*","*","*","PipeUtilization_*.csv")):
        for row in csv.DictReader(open(p)):
            for k,col in (("aiv_time","aiv_time(us)"),("vec","aiv_vec_time(us)"),
                          ("mte2","aiv_mte2_time(us)"),("mte3","aiv_mte3_time(us)"),
                          ("scalar","aiv_scalar_time(us)")):
                x=row.get(col)
                if x not in (None,"","NA"): eng[k].append(float(x))
    mem={k:[] for k in ("gm_to_ub_kb","ub_to_gm_kb")}
    for p in glob.glob(os.path.join(d,"OPPROF_*","*","*","Memory_*.csv")):
        for row in csv.DictReader(open(p)):
            for k,col in (("gm_to_ub_kb","GM_to_UB_datas(KB)"),("ub_to_gm_kb","UB_to_GM_datas(KB)")):
                x=row.get(col)
                if x not in (None,"","NA"): mem[k].append(float(x))
    out={"block_dim":blocks,"n":len(durs),
         "duration_us":round(statistics.median(durs),3) if durs else None}
    for k,vals in list(eng.items())+list(mem.items()):
        out[k]=round(statistics.median(vals),3) if vals else None
    return out

def main():
    variants = sys.argv[1:] or sorted(VARIANTS)
    for v in variants:
        od = os.path.join("profiles","copy_floor",v)
        os.makedirs(od, exist_ok=True); os.chmod(od,0o700)
        cmd=["msprof","op","--kernel-name=main",f"--output={od}",
             f"--launch-count={LAUNCH_COUNT}",f"--warm-up={WARM_UP}","--dump=off",
             f"--aic-metrics={METRICS}", sys.executable,"copy_floor2.py","--case",v,
             "--iters",str(WARM_UP+LAUNCH_COUNT)]
        lg=os.path.join("logs","copy_floor"); os.makedirs(lg,exist_ok=True)
        with open(os.path.join(lg,v+".log"),"w") as f:
            f.write(" ".join(cmd)+"\n"); f.flush()
            pr=subprocess.run(cmd,capture_output=True,text=True); f.write(pr.stdout+pr.stderr)
        if pr.returncode!=0:
            print(f"{v}: msprof rc={pr.returncode}"); continue
        res=parse(od)
        if res["duration_us"] is None:
            print(f"{v}: no rows"); continue
        rd,wr,units=expected(v)
        shape,ch,rr,wr_rows,n_units,kind,rs,ws=spans(v)
        dur=res["duration_us"]
        rec={"variant":v,"group":"dma_ceiling","shape":SHAPES[shape],"shape_key":shape,
             "dtype":"float16","ch":ch,"read_rows":rr,"write_rows":wr_rows,
             "read_span_elems":rs,"write_span_elems":ws,"kind":kind,"units":units,
             "expected_read_bytes":rd,"expected_write_bytes":wr,
             "duration_us":dur,
             "achieved_total_bw_gbs":round((rd+wr)/dur/1e3,1),
             "achieved_read_bw_gbs":round(rd/dur/1e3,1) if rd else None,
             "achieved_write_bw_gbs":round(wr/dur/1e3,1) if wr else None,
             **res,"msprof_raw_path":od,"timestamp":TS}
        with open(OUT,"a") as f: f.write(json.dumps(rec)+"\n")
        print(f"{v:16s} dur={dur:8.2f} rd_span={rs*2:6d}B wr_span={ws*2:6d}B "
              f"rd={rd/1e6:6.2f}MB wr={wr/1e6:6.2f}MB "
              f"tot={rec['achieved_total_bw_gbs']:7.1f} rd={rec['achieved_read_bw_gbs']} "
              f"wr={rec['achieved_write_bw_gbs']} mte2={res['mte2']} mte3={res['mte3']} "
              f"vec={res['vec']} blk={res['block_dim']}")

if __name__=="__main__":
    main()
