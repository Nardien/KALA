import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import transformers

from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter
from models.layers.gatv3_conv import GATv3Conv

class KFM(nn.Module):
    def __init__(self, args, layer_num, entity_embed=None, relation_embed=None):
        super(KFM, self).__init__()

        if type(entity_embed) == list:
            self.entity_embed = torch.nn.Embedding(
                len(entity_embed)+1,
                args.entity_embed_size, 
                padding_idx=0, 
            )
        else:
            self.entity_embed = torch.nn.Embedding.from_pretrained(
                entity_embed,
                padding_idx=0,
                freeze=False,
            )

        if entity_embed is not None:
            self.entity_embed.requires_grad = True

        self.relation_embed = torch.nn.Embedding(
            100,
            128,
        )
        if relation_embed is not None:
            self.relation_embed.requires_grad = True

        self.kfms = nn.ModuleList(
            [KFMLayer(args, self.entity_embed, self.relation_embed) for _ in range(layer_num)]
        )

    def forward(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.kfms[idx]

class KFMLayer(nn.Module):
    def __init__(self, args, entity_embed, relation_embed):
        super(KFMLayer, self).__init__()
        self.loc_layer = args.loc_layer
        self.hidden_size = args.hidden_size
        self.graph_hidden_size = args.entity_embed_size

        self.entity_embed = entity_embed
        self.relation_embed = relation_embed
        self.device = args.device
        
        """ Graph Neural Network """
        self.gnn = GNNLayer(args)

        """ Gamma, Beta """
        self.pre_weight_linear = nn.Sequential(
            nn.Linear(self.graph_hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.pre_bias_linear = nn.Sequential(
            nn.Linear(self.graph_hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.post_weight_linear = nn.Sequential(
            nn.Linear(self.graph_hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.post_bias_linear = nn.Sequential(
            nn.Linear(self.graph_hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

    def forward(self, 
        hidden_states, # h
        mention_positions,
        nodes, # graph.x
        edge_index, # graph.edge_index
        graph_batch, # graph.batch
        edge_attr,
        local_indicator,
    ):
        B, _, D = hidden_states.shape
        """ scatter hidden states follwing mention positions """
        scatter_states = scatter(hidden_states, mention_positions+1, dim=1)[:,1:,:]
        # Stable scatter mask
        scatter_mask = (scatter_states.sum(-1).float() != 0.0)
        scatter_states = scatter_states[scatter_mask]
        all_states = torch.zeros(nodes.shape[0], scatter_states.shape[-1]).to(device=nodes.device)
        all_states[local_indicator.bool()] = scatter_states
        all_states = all_states.detach()

        """ Encode Graph"""
        node_embeds = self.entity_embed(nodes)
        if len(node_embeds) == 0:
            graph_embeds = torch.zeros_like(hidden_states)
            graph_mask = torch.zeros_like(hidden_states)
        else:
            if self.relation_embed is not None:
                rel_embeds = self.relation_embed(edge_attr)
            else:
                rel_embeds = None
            mask = nodes == 0

            graph_embeds = self.gnn(node_embeds, edge_index, rel_embeds, 
                                    mask=mask, hidden_states=all_states)
            graph_embeds, _ = to_dense_batch(graph_embeds, graph_batch)

            _B, L, _ = graph_embeds.shape
            # Sometimes, the last batch may not be appended
            while _B < B:
                graph_embeds = torch.cat([graph_embeds, torch.zeros((1, L, D), device=self.device)],
                                        dim=0)
                _B, L, D = graph_embeds.shape

            # Zero padding in front to cover NO corresponding node case
            graph_embeds = torch.cat([torch.zeros((B, 1, D), device=self.device),
                                    graph_embeds], dim=1)
            mention_positions = mention_positions + 1 # Shift

            # To one-hot
            N = graph_embeds.shape[1] # Dense nubmer of nodes
            onehot_node_ids = torch.eye(N).to(self.device)[mention_positions]
            graph_embeds = torch.matmul(onehot_node_ids, graph_embeds)
            graph_mask = (mention_positions != 0).to(self.device).unsqueeze(-1).float()

        pre_gamma = self.pre_weight_linear(graph_embeds) * graph_mask
        pre_beta = self.pre_bias_linear(graph_embeds) * graph_mask
        post_gamma = self.post_weight_linear(graph_embeds) * graph_mask
        post_beta = self.post_bias_linear(graph_embeds) * graph_mask
        return pre_gamma, pre_beta, post_gamma, post_beta

class GNNLayer(nn.Module):
    def __init__(self, args):
        super(GNNLayer, self).__init__()
        num_gnn_layers = args.num_gnn_layers
        self.dropout = nn.Dropout(0.1)
        self.convs = nn.ModuleList()
        for layer in range(num_gnn_layers):
            self.convs.append(
                GATv3Conv(args.entity_embed_size, args.entity_embed_size, rel_embed_size=128)
            )

    def forward(self, x, edge_index, edge_attr=None, hidden_states=None, mask=None):
        for i, conv in enumerate(self.convs):
            if i == 0:
                x = conv(x, edge_index, edge_attr, mask=mask, hidden_states=hidden_states)
            else:
                x = conv(x, edge_index, edge_attr, mask=None, hidden_states=hidden_states)
            x = self.dropout(F.relu(x))
        return x
