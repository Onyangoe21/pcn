import os
import torch
import torch.optim as optim
from data.preprocess_connectome import preprocess_connectome
from lib.helpers.trainer import Trainer
from lib.helpers.model_helpers import get_complex_model

# Load dataset
data_folder = "connectome/"
graphs = preprocess_connectome(data_folder)

# Train/Test Split (80/20)
train_size = int(0.8 * len(graphs))
train_data = graphs[:train_size]
test_data = graphs[train_size:]

# Model Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = {
    "batch_size": 32,
    "lr": 0.001,
    "epochs": 100,
    "num_layers": 4,
    "model": "sparse_cin",
    "dropout": 0.4,
}

# Initialize Model
model = get_complex_model(args, train_data, device)
optimizer = optim.Adam(model.parameters(), lr=args["lr"])

# Train Model
trainer = Trainer(model, args, train_data, test_data, optimizer)
trainer.train()
