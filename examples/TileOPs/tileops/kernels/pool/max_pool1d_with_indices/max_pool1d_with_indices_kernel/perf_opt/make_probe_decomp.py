"""Cost-decomposition probes for the baseline kernel (timing only, results wrong).

Each probe removes exactly one component of the inner loop so that the marginal
cost can be read off `msprof op`.  These are NOT candidates and never enter
perf_records.jsonl.  Raw profiles land in profiles/probe/<variant>/.

    python make_probe_decomp.py            # writes ../perf_opt/_probe_*.py
"""

import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "_max_pool1d_with_indices_kernel.py")
HERE = os.path.dirname(os.path.abspath(__file__))

PASS2 = """                            # Pass 2: first-hit.  The value accumulator is final,
                            # so eq(row, acc) marks every lane whose value equals
                            # the window maximum; vmin then keeps the smallest iw.
                            for kw in T.serial(kernel_w):
                                for owl in T.serial(bl):
                                    iw = base + owl * stride_w + kw * dilation_w
                                    j = iw - s0
                                    if 0 <= iw and iw < l_in:
                                        T.vbrc(T.int32(iw), fvec)
                                        T.vcmp(
                                            row_f32[j:j + 1, 0:ch],
                                            acc[owl:owl + 1, 0:ch],
                                            eqb,
                                            "eq",
                                        )
                                        T.vselect(eqb, fvec, big1, cand)
                                        T.vmin(
                                            max_idx[owl:owl + 1, 0:ch],
                                            cand,
                                            max_idx[owl:owl + 1, 0:ch],
                                        )
"""

NAN_PASS1 = """                                        T.vbrc(T.int32(iw), fvec)
                                        # last-NaN: non-NaN -> -1, NaN -> iw, then
                                        # vmax accumulates the largest NaN iw.
                                        T.vcmp(
                                            row_f32[j:j + 1, 0:ch],
                                            row_f32[j:j + 1, 0:ch],
                                            eq2b,
                                            "eq",
                                        )
                                        T.vselect(eq2b, neg1, fvec, nanc)
                                        T.vmax(
                                            nan_idx[owl:owl + 1, 0:ch],
                                            nanc,
                                            nan_idx[owl:owl + 1, 0:ch],
                                        )
"""

INT_CMP = """                            T.vcmp(nan_idx, zeros_i, cfinal, "ge")
                            T.vselect(cfinal, nanbuf, acc, out32)
                            T.vselect(cfinal, nan_idx, max_idx, idx32)"""
INT_CMP_SAFE = """                            T.vbrc(neg1_i, nan_idx)
                            T.vbrc(zero_i, idx32)
                            T.copy(max_idx, out32)"""
EMPTY = """                            T.vcmp(idx32, bigv, cfinal, "eq")
                            T.vselect(cfinal, zeros_i, idx32, idx32)"""


def write(name: str, src: str) -> None:
    path = os.path.join(HERE, f"_probe_{name}.py")
    open(path, "w").write(src)
    print(f"wrote {path}")


def main():
    src = open(BASE).read()
    assert src.count(PASS2) == 1 and src.count(NAN_PASS1) == 1
    assert src.count(INT_CMP) == 1 and src.count(EMPTY) == 1

    # P0: baseline copy (harness self-check)
    write("decomp_base", src)

    # P1: no pass 2 (no first-hit scan at all)
    write("decomp_nopass2", src.replace(PASS2, ""))

    # P2: no NaN tracking inside pass 1
    write("decomp_nonantrack", src.replace(NAN_PASS1, ""))

    # P3: no per-unit int32 vcmp epilogue
    write("decomp_nointcmp",
          src.replace(INT_CMP, INT_CMP_SAFE).replace(EMPTY, ""))

    # P4: neither pass 2 nor NaN tracking
    write("decomp_min", src.replace(PASS2, "").replace(NAN_PASS1, ""))

    # P5: dead-code removal only (reference for R1b)
    dead = src.replace(
        """                    acc = T.alloc_ub((bl, ch), "float32")
                    flag = T.alloc_ub((bl, ch), "float32")""",
        """                    acc = T.alloc_ub((bl, ch), "float32")""")
    dead = dead.replace(
        """                    eq_mask = T.alloc_ub((lspan, ch), "bool")
                    eq_f32 = T.alloc_ub((lspan, ch), "float32")""", "")
    dead = dead.replace("                            T.vbrc(one_val, flag)\n", "")
    dead = dead.replace(
        """                            T.vcmp(row_f32, row_f32, eq_mask, "eq")
                            T.vcast(eq_mask, eq_f32, round_mode="rint")
""", "")
    dead = dead.replace(
        """                                        T.vmul(
                                            flag[owl:owl + 1, 0:ch],
                                            eq_f32[j:j + 1, 0:ch],
                                            flag[owl:owl + 1, 0:ch],
                                        )
""", "")
    dead = dead.replace("                    one_val = T.float32(1.0)\n", "")
    write("decomp_deadcode", dead)

    # P6: int32 vcmp kept but pass 2 removed (int-cmp isolated on top of P1)
    write("decomp_nopass2_intcmp",
          src.replace(PASS2, "").replace(EMPTY, ""))


if __name__ == "__main__":
    main()
