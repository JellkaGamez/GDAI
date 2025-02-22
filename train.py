import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import GDTransformer  # Assuming your model is in model.py

# Hyperparameters
EPOCHS = 3  # Number of times to train
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DATA_FOLDER = "data"
MODEL_PATH = "gd_model.pth"

# Custom Dataset Loader
class GDDataset(Dataset):
    def __init__(self, data_folder):
        self.data = []
        for file in os.listdir(data_folder):
            if file.endswith(".json"):
                with open(os.path.join(data_folder, file), "r") as f:
                    self.data.extend(json.load(f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obj = self.data[idx]
        obj_tokens = torch.tensor(obj["tokens"], dtype=torch.long)
        obj_attrs = torch.tensor(obj["attributes"], dtype=torch.float)
        target = torch.tensor(obj["next_token"], dtype=torch.long)
        return obj_tokens, obj_attrs, target

# Load Dataset
dataset = GDDataset(DATA_FOLDER)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GDTransformer(vocab_size=2000, embed_dim=128, num_heads=8, num_layers=6, audio_dim=44100, hidden_dim=512, num_classes=2000).to(device)

# Load existing model if available
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model.load_state_dict(torch.load(MODEL_PATH))

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS}...")
    for obj_tokens, obj_attrs, target in dataloader:
        obj_tokens, obj_attrs, target = obj_tokens.to(device), obj_attrs.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(obj_tokens, obj_attrs, torch.randn(obj_tokens.shape[0], 1, 44100).to(device))  # Random waveform data for now
        loss = criterion(output.view(-1, output.shape[-1]), target.view(-1))
        loss.backward()
        optimizer.step()

    # Save Model & Backup
    torch.save(model.state_dict(), MODEL_PATH)
    torch.save(model.state_dict(), f"backup_model_epoch{epoch}.pth")
    print(f"Saved model backup at epoch {epoch}.")

print("Training complete!")
