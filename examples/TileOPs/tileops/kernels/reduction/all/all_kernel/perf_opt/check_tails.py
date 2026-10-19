import importlib.util, os, sys
import torch, torch_npu
os.environ.setdefault("TILELANG_ASCEND_MODE","Developer")
def load(p):
    n="m_"+os.path.basename(os.path.dirname(os.path.abspath(p)))+"_"+os.path.basename(p).replace(".","_")
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m
path=sys.argv[1]
m=load(os.path.abspath(path))
for N in (8193, 8200, 8256, 8320, 9000):
    cfg=m._select_config(32,N,"float16")
    x=m._make_input(32,N,"float16")
    k=m._logical_reduce_kernel(32,N,"all","float16")(cfg["block_m"])
    y=k(x).bool().cpu(); ref=torch.all(x.cpu(),dim=-1)
    print(f"N={N} cfg=({cfg['num_full']},{cfg['tail']}) equal={torch.equal(y,ref)} bad={int((y!=ref).sum())}")
