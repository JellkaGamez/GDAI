import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ==== HYPERPARAMETERS ====
EPOCHS = 1
BATCH_SIZE = 2
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
# ==== CUSTOM DATASET ====
class GDDataset(Dataset):
    def __init__(self, data_folder, pad_value=0):
        self.data = self.load_data(data_folder)  # Load JSON data
        self.pad_value = pad_value
        self.max_length = max(len(obj) for obj in self.data) if self.data else 1

    def load_data(self, folder):
        """Loads and processes JSON files from a folder into usable token sequences."""
        all_sequences = []
        print(f"Loading data from folder: {folder}")
        print(f"Files in folder: {os.listdir(folder)}")
        for filename in os.listdir(folder):
            if filename.endswith(".json"):
                print(f"Processing file: {filename}")
                with open(os.path.join(folder, filename), "r", encoding="utf-8") as file:
                    try:
                        level_data = json.load(file)  # Load JSON
                        # print(f"Loaded data from {filename}: {level_data}")  # Debug print
                        tokens = self.process_data(level_data)  # Convert to tokens
                        if tokens:  # Ensure valid sequence
                            all_sequences.append(tokens)
                        else:
                            print(f"⚠️ Warning: Empty sequence in file: {filename}")
                    except json.JSONDecodeError:
                        print(f"⚠️ Warning: Skipping invalid JSON file: {filename}")
        if not all_sequences:
            print("🚨 Warning: No valid sequences found!")
        return all_sequences

    def process_data(self, level_data):
        """Converts a Geometry Dash level JSON into a list of token IDs."""
        tokens = []
        # print(f"Processing level data: {level_data}")  # Debug print
        for obj in level_data:  # Assume level_data is a list of objects
            if isinstance(obj, list):  # Check if the object is a list
                # Extract the ID from the first element of the list
                obj_id = obj[0].split(":")[1]  # Extract the ID from "id:1006"
                tokens.append(int(obj_id))  # Append object ID as a token
            else:
                print(f"⚠️ Warning: Unexpected object format: {obj}")
        # print(f"Tokens extracted: {tokens}")  # Debug print
        return tokens

    def __getitem__(self, idx):
        token_ids = self.data[idx]
        # print(f"Sequence at index {idx}: {token_ids}")  # Debug print

        # Handle short sequences
        if len(token_ids) < 2:
            print(f"🚨 Warning: Sequence too short. Padding with dummy data.")
            token_ids = [0, 0]  # Use dummy data for short sequences

        # Padding the sequence to max_length if necessary
        if len(token_ids) < self.max_length:
            token_ids = token_ids + [self.pad_value] * (self.max_length - len(token_ids))
        
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
print(f"Dataset size: {len(dataset)}")
print(f"Sample data: {dataset[0]}")  # Print the first sequence

# Initialize DataLoader
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

    for input_seq in dataloader:
        input_seq = input_seq.to(device)

        # Prepare the target by shifting the sequence
        target = input_seq[:, 1:]  # Remove the first token
        input_seq = input_seq[:, :-1]  # Remove the last token for input
        
        optimizer.zero_grad()
        output = model(input_seq)
        
        if output.shape[1] == 0:  # 🔥 Check if sequence is empty
            print("Warning: Empty output sequence, skipping...")
            continue
        
        # We are predicting the last token of the sequence (shifted target)
        loss = criterion(output.view(-1, vocab_size), target.contiguous().view(-1))

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
