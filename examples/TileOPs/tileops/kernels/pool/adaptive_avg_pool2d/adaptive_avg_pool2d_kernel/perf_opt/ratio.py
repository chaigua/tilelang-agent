#!/usr/bin/env python3
"""Compute the msprof roofline Ratio for one profiled run, in the same 口径 as
`tileops/benchmark/msprof_roofline.py` (Method 2, CSV formulas).

  python3 ratio.py <profile_run_dir | profile_out_dir>

Formulas (compute-roofline):
  total_fops  = sum(aiv_vec_fops)                          (ArithmeticUtilization)
  total_bytes = (L2 read miss) * cacheline(128)             (L2Cache)
  AI          = total_fops / total_bytes
  perf        = total_fops / duration
  computility = (64*fp32_prop + 32*misc_prop) * freq*1e6 * num_cores
  roofline    = min(AI * gm_bw(1.8 TB/s), computility)
  ratio       = perf / roofline

num_cores defaults to 40 (910B3 vector cores actually used by this family;
validated against the Stage-5 report: resnet final 16.7203us -> Ratio 2.9020%,
AI 5.4772, Perf 0.1315 TOps/s, Computility 4.5326 TOps/s).
"""
import csv
import glob
import statistics
import os
import sys

CACHELINE = 128
GM_BW = 1.8  # TB/s
NUM_CORES = 40


def sf(v):
    if v in (None, "", "NA"):
        return 0.0
    try:
        return float(v)
    except ValueError:
        return 0.0


def newest_run_dir(d):
    if glob.glob(os.path.join(d, "OpBasicInfo*.csv")):
        return d
    subs = [s for s in glob.glob(os.path.join(d, "OPPROF_*", "*", "*")) if glob.glob(os.path.join(s, "OpBasicInfo*.csv"))]
    return sorted(subs)[-1] if subs else None


def read_rows(d, pat):
    p = sorted(glob.glob(os.path.join(d, pat)))
    if not p:
        return []
    with open(p[-1]) as f:
        return list(csv.DictReader(f))


def main():
    run = newest_run_dir(sys.argv[1].rstrip("/"))
    if run is None:
        print("no OpBasicInfo found under", sys.argv[1])
        return
    # duration = MEDIAN over all OpBasicInfo rows of this run (the kernel-only
    # latency metric), so the Ratio denominator matches the reported Task Duration.
    _obs = [sf(r["Task Duration(us)"]) for r in read_rows(run, "OpBasicInfo*.csv")]
    _obs = [d for d in _obs if d > 0]
    ob = read_rows(run, "OpBasicInfo*.csv")[0]
    duration = statistics.median(_obs) if _obs else sf(ob["Task Duration(us)"])
    freq = sf(ob.get("Current Freq")) or 1800.0
    au = read_rows(run, "ArithmeticUtilization*.csv")
    vec = [r for r in au if r.get("sub_block_id", "").startswith("vector")]
    fops = sum(sf(r.get("aiv_vec_fops")) for r in vec)
    svec = sum(sf(r.get("aiv_vec_ratio")) * sf(r.get("aiv_total_cycles")) for r in vec)
    sfp32 = sum(sf(r.get("aiv_vec_fp32_ratio")) * sf(r.get("aiv_total_cycles")) for r in vec)
    smisc = sum(sf(r.get("aiv_vec_misc_ratio")) * sf(r.get("aiv_total_cycles")) for r in vec)
    fp32p = sfp32 / svec if svec else 0.0
    miscp = smisc / svec if svec else 0.0
    peak = 64 * fp32p + 32 * miscp
    l2 = read_rows(run, "L2Cache*.csv")
    miss = sum(sf(r.get("aiv_r0_read_cache_miss_allocate")) + sf(r.get("aiv_r1_read_cache_miss_allocate")) for r in l2)
    miss += sum(sf(r.get("aic_r0_read_cache_miss_allocate")) + sf(r.get("aic_r1_read_cache_miss_allocate")) for r in l2)
    bytes_ = miss * CACHELINE
    perf = fops / (duration * 1e-6) / 1e12 if duration else 0.0
    ai = fops / bytes_ if bytes_ else 0.0
    comp = peak * freq * 1e6 * NUM_CORES / 1e12
    rl = min(ai * GM_BW, comp)
    ratio = perf / rl if rl else 0.0
    print(f"{run}")
    print(f"  duration={duration:.4f}us freq={freq:.0f}MHz vec_fops={fops:.0f} l2_miss={miss:.0f} bytes={bytes_:.0f}")
    print(f"  AI={ai:.4f} Ops/B  Perf={perf:.4f} TOps/s  Computility={comp:.4f} TOps/s")
    print(f"  Ratio={ratio*100:.4f}%  (num_cores={NUM_CORES}, gm_bw={GM_BW})")


if __name__ == "__main__":
    main()
