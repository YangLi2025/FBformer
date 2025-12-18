from typing import Optional

import numpy as np
import torch
from torch import nn

from torch import Tensor
from torch_scatter import gather_csr, scatter, segment_csr

from torch_geometric.utils.num_nodes import maybe_num_nodes

class RBFExpansion(nn.Module):  

    def __init__(
        self,
        vmin: float = 0, 
        vmax: float = 8, 
        bins: int = 40,
        lengthscale: Optional[float] = None, 
    ):
      
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins) 
        )

        if lengthscale is None: 
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale 
        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2) 

    def forward(self, distance: torch.Tensor) -> torch.Tensor: 
    
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2 
        )


@torch.jit.script 
def softmax(src: Tensor, index: Optional[Tensor] = None,
            ptr: Optional[Tensor] = None, num_nodes: Optional[int] = None,
            dim: int = 0) -> Tensor:

    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim 
        size = ([1] * dim) + [-1] 
        ptr = ptr.view(size) 
        src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr) 
        out = (src - src_max).exp() 
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr) 
    elif index is not None: 
        N = maybe_num_nodes(index, num_nodes) 
        src_max = scatter(src, index, dim, dim_size=N, reduce='max')  
        src_max = src_max.index_select(dim, index) 
        out = (src - src_max).exp() 
        out_sum = scatter(out, index, dim, dim_size=N, reduce='sum') 
        out_sum = out_sum.index_select(dim, index) 
    else: 
        raise NotImplementedError

    return out / (out_sum + 1e-16) 

@torch.jit.script
def softmax_vec(src: Tensor, index: Optional[Tensor] = None,
            ptr: Optional[Tensor] = None, num_nodes: Optional[int] = None,
            dim: int = 0) -> Tensor:

    if ptr is not None: 
        dim = dim + src.dim() if dim < 0 else dim 
        size = ([1] * dim) + [-1] 
        src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr) 
        out = (src - src_max).exp() 
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr) 
    elif index is not None: 
        N = maybe_num_nodes(index, num_nodes) 
        src_max = scatter(src, index, dim, dim_size=N, reduce='max') 
        src_max = src_max.index_select(dim, index) 
        out = (src - src_max).exp() 
        out_sum = scatter(out, index, dim, dim_size=N, reduce='sum') 
        out_sum = out_sum.index_select(dim, index) 
    else: 
        raise NotImplementedError

    return out / (out_sum + 1e-16) 