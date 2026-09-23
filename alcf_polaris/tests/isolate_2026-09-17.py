import sys, faulthandler, traceback
faulthandler.enable()
which = sys.argv[1]
def t_fa():
    import torch, flash_attn
    from flash_attn import flash_attn_func
    q = torch.randn(1,128,8,64, device="cuda", dtype=torch.float16)
    o = flash_attn_func(q,q,q); torch.cuda.synchronize()
    return f"flash_attn {flash_attn.__version__} out {tuple(o.shape)}"
def t_fa_after_tf():
    import tensorflow as tf, jax, jax.numpy as jnp
    tf.reduce_sum(tf.ones((8,8))).numpy(); float(jnp.ones(8).sum())
    return t_fa()
def t_te():
    import torch, transformer_engine.pytorch as te
    l = te.Linear(64,64).cuda(); y = l(torch.randn(8,64,device="cuda")); torch.cuda.synchronize()
    return f"TE linear {tuple(y.shape)}"
def t_ds():
    import deepspeed; from deepspeed.ops.adam import FusedAdam
    import torch; p = torch.nn.Parameter(torch.randn(10, device="cuda")); FusedAdam([p]).step()
    return f"deepspeed {deepspeed.__version__} FusedAdam ok"
def t_vllm():
    import vllm; return f"vllm {vllm.__version__}"
def t_compile():
    import torch
    f = torch.compile(lambda x: (x*2).sin())
    return f"torch.compile/triton {f(torch.randn(64,device='cuda')).shape}"
def t_mpi4jax():
    import jax, jax.numpy as jnp, mpi4jax
    return "mpi4jax import ok"
def t_ort():
    import onnxruntime as ort; return f"onnxruntime {ort.__version__} {ort.get_available_providers()}"
def t_xgb():
    import xgboost as xgb, numpy as np
    d = xgb.DMatrix(np.random.rand(100,4), label=np.random.rand(100))
    xgb.train({"device":"cuda","tree_method":"hist"}, d, 2); return f"xgboost {xgb.__version__} gpu hist ok"
def t_mamba():
    import torch; from mamba_ssm import Mamba
    m = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda(); y = m(torch.randn(1,16,64,device="cuda")); torch.cuda.synchronize()
    return f"mamba_ssm out {tuple(y.shape)}"
def t_flashinfer():
    import flashinfer; return f"flashinfer {flashinfer.__version__}"
def t_gce():
    import globus_compute_endpoint, parsl; return f"gce {globus_compute_endpoint.__version__} parsl {parsl.__version__}"
tests = dict(fa=t_fa, fa_after_tf=t_fa_after_tf, te=t_te, ds=t_ds, vllm=t_vllm, compile=t_compile, mpi4jax=t_mpi4jax, ort=t_ort, xgb=t_xgb, mamba=t_mamba, flashinfer=t_flashinfer, gce=t_gce)
try:
    print(f"[{which}] OK", tests[which](), flush=True)
except Exception as e:
    print(f"[{which}] FAIL {e!r}", flush=True); traceback.print_exc()
