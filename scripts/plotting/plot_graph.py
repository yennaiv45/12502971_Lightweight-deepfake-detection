import json
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION: Set the target history file ---
history_filename = 'history_with_aug.json'
# ---------------------------------------------------

# Build absolute paths from the script's location to make it runnable from anywhere
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir)) # up two levels
artifacts_path = os.path.join(project_root, 'artifacts', history_filename)
figures_path = os.path.join(project_root, 'figures')


if not os.path.exists(artifacts_path):
    print(f"❌ CRITICAL ERROR: File not found: '{artifacts_path}'")
    print("Ensure you have run the training script and the artifacts are in the correct folder.")
    exit()

print(f"✅ Loading data from: {artifacts_path}")

with open(artifacts_path, 'r') as f:
    history = json.load(f)

epochs = range(1, len(history['train_loss']) + 1)

# Create the 'figures' directory if it doesn't exist
os.makedirs(figures_path, exist_ok=True)

# 1. Plot Loss (Train vs. Val)
plt.figure(figsize=(10, 6))
plt.plot(epochs, history['train_loss'], label='Train Loss')
plt.plot(epochs, history['val_loss'], label='Val Loss')
plt.title('Model Loss Evolution (With Augmentation)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(figures_path, 'loss_curve.png')) # Overwrite if exists
print(f"Saved chart: {os.path.join(figures_path, 'loss_curve.png')}")

# 2. Plot AUC (Train vs. Val)
plt.figure(figsize=(10, 6))
plt.plot(epochs, history['train_auc'], label='Train AUC')
plt.plot(epochs, history['val_auc'], label='Val AUC')
plt.title('Model AUC Evolution (With Augmentation)')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(figures_path, 'accuracy_curve.png')) # Overwrite if exists
print(f"Saved chart: {os.path.join(figures_path, 'accuracy_curve.png')}")