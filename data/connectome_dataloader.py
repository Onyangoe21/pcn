import os
import torch
import networkx as nx
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx

class ConnectomeDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.files = [f for f in os.listdir(root) if f.endswith(".graphml")]

    def len(self):
        return len(self.files)

    def get(self, idx):
        graph_path = os.path.join(self.root, self.files[idx])
        graph = nx.read_graphml(graph_path)
        
        # Convert to PyTorch Geometric format
        pyg_data = from_networkx(graph)

        # Normalize edge weights (optional)
        if 'weight' in pyg_data.edge_attr:
            pyg_data.edge_attr = pyg_data.edge_attr / torch.max(pyg_data.edge_attr)

        return pyg_data
