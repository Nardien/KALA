from typing import Union, Tuple, Optional
from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor)
import math

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros

class GATv3Conv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: int,
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True,
                 share_weights: bool = False,
                 rel_embed_size: int = 100,
                 **kwargs):
        super(GATv3Conv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        out_channels = out_channels // heads # To make the output dimension as same
        self.out_channels = out_channels 
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights
        self.relation_type_dim = rel_embed_size

        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, heads * out_channels, bias=bias)

        self.lin_e = Linear(self.relation_type_dim, heads * out_channels, bias=bias)
        self.lin_h = Linear(in_channels, heads * out_channels, bias=False)

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.lin_e.weight)
        glorot(self.lin_h.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None, 
                return_attention_weights: bool = None,
                mask: OptTensor = None,
                hidden_states: OptTensor = None):
        # type: (Union[Tensor, PairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None

        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr=edge_attr)
                N = edge_index.shape[1]
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
                aggr_mask = torch.zeros(N, device=edge_index.device, dtype=torch.bool)
                # Zero padding to edge_attr
                N = edge_index.shape[1] - N
                loop_attr = edge_attr.new_full((N, self.relation_type_dim), 0.0)
                edge_attr = torch.cat([edge_attr, loop_attr], dim=0)
                if mask is None:
                    mask = torch.zeros(N, device=edge_index.device, dtype=torch.bool)
                aggr_mask = torch.cat([aggr_mask, mask])

            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        if edge_attr is not None:
            x_e = self.lin_e(edge_attr).view(-1, H, C)
        
        if hidden_states is not None:
            x_h = self.lin_h(hidden_states).view(-1, H, C)
            x_r = x_h + x_r
        # propagate_type: (x: PairTensor)
        # x_l -> x_j, x_r -> x_i
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=x_e, size=size, mask=aggr_mask)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out
    
    def message(self, x_j: Tensor, x_i: Tensor,
                edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int],
                mask: OptTensor) -> Tensor:
        x = x_i + x_j + edge_attr
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = alpha + (mask.view(alpha.shape) * -10000.0)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)