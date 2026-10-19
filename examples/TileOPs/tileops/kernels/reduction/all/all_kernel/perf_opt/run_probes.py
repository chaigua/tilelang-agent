"""Calibration probe runner (NOT tune workloads).

Runs the ``probe_floor.py`` modes for the tile sizes that the three tune
workloads actually use, appends the raw measurements to
``copy_floor_calibration.jsonl``, and derives the per-stage unit prices into
``op_cost_calibration.jsonl``.  Nothing here enters ``perf_records.jsonl``.

Probe shapes are *tile-shaped* synthetics (M=32 rows, W = the tile width), so
the measured cost is per tile and composable:

  * W=4096  -> 16 KB f32 tile (4k path-R tile, 32 KB at bm=2)
  * W=8192  -> 32 KB f32 tile (32k column tile)
  * W=16384 -> 64 KB f32 tile (3d / 32k R'-scale tile)

CLI: python run_probes.py [--launch-count 15] [--modes empty,copyout,read,copy]
"""

import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_bench import measure  # noqa: E402

PROBE_IMPL = os.path.join(HERE, "probe_floor.py")
COPY_FLOOR = os.path.join(HERE, "copy_floor_calibration.jsonl")
OP_COST = os.path.join(HERE, "op_cost_calibration.jsonl")

# (M, N, bm) probe grid: W x bm x 4 B = the f32 tile the kernel actually moves
GRID = [
    (32, 4096, 1),
    (32, 4096, 2),
    (32, 4096, 4),
    (32, 8192, 1),
    (32, 16384, 1),
]


def append(path, row):
    with open(path, "a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch-count", type=int, default=15)
    ap.add_argument("--warm-up", type=int, default=3)
    ap.add_argument("--modes", default="empty,copyout,read,copy")
    ap.add_argument("--grid", default=None, help="semicolon list M,N,bm")
    args = ap.parse_args()

    grid = GRID
    if args.grid:
        grid = [tuple(int(v) for v in item.split(",")) for item in args.grid.split(";")]
    modes = [m for m in args.modes.split(",") if m]

    results = {}
    for (M, N, bm) in grid:
        tile_bytes = M * 0 + bm * N * 4  # f32 kernel bytes per tile
        for mode in modes:
            out_dir = os.path.join(
                "profiles", "probes", f"M{M}_N{N}_bm{bm}",
                mode,
            )
            log = os.path.join("logs", "probes", f"M{M}_N{N}_bm{bm}_{mode}.log")
            res = measure(
                PROBE_IMPL, "probe", out_dir,
                kernel_name="main", launch_count=args.launch_count,
                warm_up=args.warm_up, log_path=log, probe=mode, block_m=bm,
                extra_args=("--probe-shape", f"{M},{N}"),
            )
            if not res.get("ok"):
                print(f"probe {mode} M={M} N={N} bm={bm}: FAILED "
                      f"{res.get('stderr_tail', '')[-200:]}", flush=True)
                append(COPY_FLOOR, {
                    "ts": datetime.datetime.now(datetime.timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"),
                    "kind": "probe", "mode": mode, "M": M, "N": N, "bm": bm,
                    "tile_bytes_f32": tile_bytes, "ok": False,
                })
                continue
            op = res["ops"]["main"]
            pipe = res.get("pipe", {}).get("main", {})
            row = {
                "ts": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"),
                "kind": "probe",
                "mode": mode,
                "M": M,
                "N": N,
                "bm": bm,
                "tile_bytes_f32": tile_bytes,
                "blocks": min((M + bm - 1) // bm, 40),
                "launch_count": args.launch_count,
                "task_duration_us_mean": round(op["mean_us"], 4),
                "median_us": round(op["median_us"], 4),
                "min_us": round(op["min_us"], 4),
                "aiv_vec_ratio": pipe.get("aiv_vec_ratio"),
                "aiv_mte2_ratio": pipe.get("aiv_mte2_ratio"),
                "aiv_mte3_ratio": pipe.get("aiv_mte3_ratio"),
                "aiv_mte2_active_bw_gbs": pipe.get("aiv_mte2_active_bw(GB/s)"),
                "profile_dir": os.path.abspath(out_dir),
                "ok": True,
            }
            results[(mode, M, N, bm)] = row["task_duration_us_mean"]
            append(COPY_FLOOR, row)
            print(f"probe {mode:8s} M={M} N={N} bm={bm} tile={tile_bytes//1024}KB: "
                  f"{op['mean_us']:.3f} us (median {op['median_us']:.3f}) "
                  f"vec={row['aiv_vec_ratio']} mte2={row['aiv_mte2_ratio']} "
                  f"bw={row['aiv_mte2_active_bw_gbs']}", flush=True)

    # --- derived unit prices -------------------------------------------------
    def get(mode, M, N, bm):
        return results.get((mode, M, N, bm))

    def add_price(**kw):
        kw["ts"] = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        kw["kind"] = "derived_price"
        kw["source"] = "copy_floor_calibration.jsonl (same run)"
        append(OP_COST, kw)
        print("price:", {k: v for k, v in kw.items() if k != "source"}, flush=True)

    for (N, bm) in [(4096, 1), (4096, 2), (4096, 4), (8192, 1), (16384, 1)]:
        empty = get("empty", 32, N, bm)
        copyout = get("copyout", 32, N, bm)
        read = get("read", 32, N, bm)
        copy = get("copy", 32, N, bm)
        if empty is None or copyout is None:
            continue
        elems = bm * N
        blocks = min((32 + bm - 1) // bm, 40)
        tasks_per_core = -(-((32 + bm - 1) // bm) // blocks)
        read_bytes = elems * 4
        add_price(
            item="mte2_read_tile",
            tile_elems=elems, tile_bytes_f32=read_bytes, bm=bm, W=N,
            blocks=blocks, tasks_per_block=tasks_per_core,
            empty_us=empty, copyout_us=copyout,
            mte2_delta_us=round(copyout - empty, 4),
            per_core_gbs=round(read_bytes / (copyout - empty) / 1e3, 3)
            if copyout > empty else None,
            aggregate_gbs=round(read_bytes * blocks / (copyout - empty) / 1e3, 3)
            if copyout > empty else None,
        )
        if read is not None:
            add_price(
                item="reduce_pass",
                tile_elems=elems, tile_bytes_f32=read_bytes, bm=bm, W=N,
                read_us=read, copyout_us=copyout,
                reduce_delta_us=round(read - copyout, 4),
            )
        if copy is not None:
            add_price(
                item="ub_copy_1to1",
                tile_elems=elems, tile_bytes_f32=read_bytes, bm=bm, W=N,
                copy_us=copy, copyout_us=copyout,
                ub_copy_delta_us=round(copy - copyout, 4),
            )


if __name__ == "__main__":
    main()
