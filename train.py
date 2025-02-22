import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ==== HYPERPARAMETERS ====
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DATA_FOLDER = "data"
FINAL_MODEL_PATH = "models/finals/gd_model.pth"
BACKUP_MODEL_PATH = "models/backups/gd_model_epoch"

# ==== ENSURE DIRECTORIES EXIST ====
os.makedirs("models/finals", exist_ok=True)
os.makedirs("models/backups", exist_ok=True)

# ==== TRANSFORMER MODEL ====
class GDTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, num_classes):
        super(GDTransformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = self.transformer_encoder(x)  # (batch, seq_len, embed_dim)
        x = self.fc_out(x)  # (batch, seq_len, vocab_size)
        return x

# ==== CUSTOM DATASET ====
class GDDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, pad_value=0):
        self.data = self.load_data(data_folder)  # Load JSON data
        self.pad_value = pad_value
        self.max_length = max(len(obj) for obj in self.data) if self.data else 1

    def load_data(self, folder):
        """Loads and processes JSON files from a folder into usable token sequences."""
        all_sequences = []
        for filename in os.listdir(folder):
            if filename.endswith(".json"):
                with open(os.path.join(folder, filename), "r", encoding="utf-8") as file:
                    try:
                        level_data = json.load(file)  # Load JSON
                        tokens = self.process_data(level_data)  # Convert to tokens
                        if tokens:  # Ensure valid sequence
                            all_sequences.append(tokens)
                    except json.JSONDecodeError:
                        print(f"⚠️ Warning: Skipping invalid JSON file: {filename}")
        return all_sequences

    def process_data(self, level_data):
        """Converts a Geometry Dash level JSON into a list of token IDs."""
        tokens = []
        for obj in level_data:  # Assume level_data["objects"] contains the level elements
            if isinstance(obj, dict) and "id" in obj:  # Check if object has an ID
                obj_id = obj["id"]
                tokens.append(obj_id)  # Append object ID as a token
        return tokens

    def __getitem__(self, idx):
        token_ids = self.data[idx]

        # Handle short sequences
        max_attempts = 10
        attempts = 0
        while len(token_ids) < 2 and attempts < max_attempts:
            idx = (idx + 1) % len(self.data)
            token_ids = self.data[idx]
            attempts += 1

        if len(token_ids) < 2:
            print(f"🚨 Warning: Could not find a valid sequence. Returning dummy data.")
            token_ids = [0, 0]

        return torch.tensor(token_ids, dtype=torch.long)

    def __len__(self):
        return len(self.data)



# ==== TRAINING SETUP ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
vocab_size = 2000  # Adjust based on actual token count
embed_dim = 128
num_heads = 8
num_layers = 6
hidden_dim = 512
num_classes = vocab_size  # Predicts the next token

# Load dataset
dataset = GDDataset(DATA_FOLDER)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
model = GDTransformer(vocab_size, embed_dim, num_heads, num_layers, hidden_dim, num_classes).to(device)

# Load pre-existing model if available
if os.path.exists(FINAL_MODEL_PATH):
    print("Loading existing model from models/finals/...")
    model.load_state_dict(torch.load(FINAL_MODEL_PATH))

# Loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==== TRAINING LOOP ====
for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS}...")
    total_loss = 0

    for input_seq, target in dataloader:
        input_seq, target = input_seq.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(input_seq)
        print(f"Output shape: {output.shape}")  # Debugging line
        if output.shape[1] == 0:  # 🔥 Check if sequence is empty
            print("Warning: Empty output sequence, skipping...")
            continue
        loss = criterion(output[:, -1, :], target)  # Predict last token

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} Loss: {avg_loss:.4f}")

    # Save Model & Backup
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    torch.save(model.state_dict(), f"{BACKUP_MODEL_PATH}{epoch}.pth")
    print(f"Saved model in models/finals/ and backup in models/backups/ (epoch {epoch}).")

print("Training complete!")
