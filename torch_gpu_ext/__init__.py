import os
import torch


this_dir = os.path.dirname(__file__)
package_name = os.path.basename(this_dir)
filename = os.path.join(os.path.dirname(this_dir), f"lib{package_name}.so")
print("Loading extension from:", filename)
torch.ops.load_library(filename)


MyObject = torch.classes.torch_gpu_ext.MyObject

fused_rope_rms = eval(f"torch.ops.{package_name}.fused_rope_rms")

@torch.library.register_fake("torch_gpu_ext::fused_rope_rms")
def fused_rope_rms_fake(
    qkv: torch.Tensor, 
    qw: torch.Tensor, 
    kw: torch.Tensor, 
    cos_sin: torch.Tensor, 
    positions: torch.Tensor, 
    num_tokens: int, 
    num_heads_q: int, 
    num_heads_k: int, 
    num_heads_v: int, 
    head_size: int, 
    is_neox_style: bool, 
    eps: float) -> None:
    return
