import os
import base64

# Path to the Sinamin Dash GDPS save file
save_file_path = os.path.expandvars(r"%LOCALAPPDATA%/SinaminDash/CCLocalLevels.dat") # Windows
# Normal GD Save file (to be made default)
# save_file_path = os.path.expandvars(r"%LOCALAPPDATA%/GeometryDash/CCLocalLevels.dat") # Windows

# XOR decryption/encryption key
XOR_KEY = 11

def xor_decrypt(data):
    """Decrypt data using XOR with the key."""
    return bytes([b ^ XOR_KEY for b in data])

def xor_encrypt(data):
    """Encrypt data using XOR with the key."""
    return bytes([b ^ XOR_KEY for b in data])

def decode_save_file(file_path):
    """Decode and decrypt the save file."""
    with open(file_path, "rb") as f:
        encoded_data = f.read()
    decoded_data = base64.b64decode(encoded_data)
    decrypted_data = xor_decrypt(decoded_data)
    return decrypted_data.decode("utf-8")

def encode_save_file(data, file_path):
    """Encrypt and encode the save file."""
    encrypted_data = xor_encrypt(data.encode("utf-8"))
    encoded_data = base64.b64encode(encrypted_data)
    with open(file_path, "wb") as f:
        f.write(encoded_data)

def add_level_to_save(save_data, level_data):
    """Add a new level to the save file."""
    # Find the position to insert the new level
    start_marker = "<k>k4</k><d>"
    end_marker = "</d></d>"
    
    # Insert the new level
    level_entry = f"<k>k4</k><d>{level_data}</d>"
    save_data = save_data.replace(end_marker, f"{level_entry}{end_marker}")
    
    return save_data

# Example: Load the generated level from a JSON file
level_name = "Generated_Level"
with open(f"levels/json/{level_name}.json", "r") as f:
    level_json = json.load(f)

# Convert the level JSON to the save file format
level_data = json.dumps(level_json)

# Decode the save file
save_data = decode_save_file(save_file_path)

# Add the new level to the save file
save_data = add_level_to_save(save_data, level_data)

# Encode and save the modified save file
encode_save_file(save_data, save_file_path)

print(f"Level '{level_name}' added to the save file at {save_file_path}")