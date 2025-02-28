import torch
import os
from connectome_dataloader import ConnectomeDataset

def preprocess_connectome(data_folder):
    dataset = ConnectomeDataset(root=data_folder)
    processed_data = []

    for i in range(len(dataset)):
        graph = dataset.get(i)
        
        # Example: Normalize node features
        if graph.x is not None:
            graph.x = (graph.x - graph.x.mean()) / (graph.x.std() + 1e-6)

        processed_data.append(graph)

    return processed_data

if __name__ == "__main__":
    data_folder = "connectome/"
    graphs = preprocess_connectome(data_folder)
    print(f"Loaded {len(graphs)} connectome graphs.")
