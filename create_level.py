import json
import torch
from torch.utils.data import DataLoader
from train import GDTransformer, GDDataset  # Assuming your model and dataset are defined in train.py

# Load the trained model
model_path = "models/finals/gd_model.pth"
vocab_size = 2000  # Adjust based on your token count
embed_dim = 128
num_heads = 8
num_layers = 6
hidden_dim = 512
num_classes = vocab_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GDTransformer(vocab_size, embed_dim, num_heads, num_layers, hidden_dim, num_classes).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Generate 100 objects
def generate_objects(model, start_token, max_length=100):
    model.eval()
    generated_objects = [start_token]
    with torch.no_grad():
        for _ in range(max_length):
            input_seq = torch.tensor([generated_objects], dtype=torch.long).to(device)
            output = model(input_seq)
            next_token = torch.argmax(output[:, -1, :], dim=-1).item()
            generated_objects.append(next_token)
    return generated_objects

# Example: Generate 100 objects starting with a specific token
start_token = 1  # Replace with your start token
generated_objects = generate_objects(model, start_token, max_length=100)

# Convert generated tokens into GD level format
def tokens_to_gd_level(tokens):
    gd_level = []
    for token in tokens:
        # Replace this with your logic to map tokens to GD object properties
        gd_level.append({
            "id": str(token),  # Example: Map token to object ID
            "x": "0",          # Example: Default x position
            "y": "0",          # Example: Default y position
            "null": "1"        # Example: Default properties
        })
    return gd_level

gd_level = tokens_to_gd_level(generated_objects)

# Save the generated level as a JSON file
level_name = "Generated_Level"
level_json = {
    "name": level_name,
    "meta": {
        "colours": [],  # Add colour data if needed
        "version": "0",
        "official-song": "0",
        "auto": "0",
        "demon": "20",
        "difficulty": "8",
        "newgrounds-song": "2",
        "length": "0",
        "placeholder-1": "0"
    },
    "length": 1000,  # Adjust based on your level length calculation
    "level": gd_level
}

with open(f"levels/json/{level_name}.json", "w") as f:
    json.dump(level_json, f, indent=4)

print(f"Generated level saved to levels/json/{level_name}.json")