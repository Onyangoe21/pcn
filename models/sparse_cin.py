import torch
import torch.nn.functional as F
from torch.nn import Linear, Embedding
from lib.layers.cin_conv import SparseCINConv
from lib.data.complex import ComplexBatch
from lib.layers.pooling import pool_complex
from lib.layers.non_linear import get_nonlinearity
from lib.layers.norm import get_graph_norm
from lib.layers.reduce_conv import InitReduceConv
from torch_geometric.nn import JumpingKnowledge

class SparseCIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=4, dropout=0.5):
        super(SparseCIN, self).__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_dim = num_features if i == 0 else hidden_dim
            self.convs.append(SparseCINConv(in_dim, hidden_dim))

        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)

