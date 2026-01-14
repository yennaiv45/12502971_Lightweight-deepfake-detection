import json
import matplotlib.pyplot as plt
import os

# --- Path Setup ---
# Build absolute paths from the script's location to make it runnable from anywhere
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir)) # up two levels
artifacts_dir = os.path.join(project_root, 'artifacts')
figures_dir = os.path.join(project_root, 'figures')

# --- Load Histories ---
try:
    with open(os.path.join(artifacts_dir, 'history_with_aug.json'), 'r') as f:
        hist_aug = json.load(f)
    with open(os.path.join(artifacts_dir, 'history_no_aug.json'), 'r') as f:
        hist_no_aug = json.load(f)
except FileNotFoundError as e:
    print(f"❌ Error loading history files: {e}")
    print("Make sure both training runs have been completed and artifacts exist.")
    exit()

# Create the 'figures' directory if it doesn't exist
os.makedirs(figures_dir, exist_ok=True)

epochs = range(1, len(hist_aug['val_loss']) + 1)

# --- Plot Loss Comparison ---
plt.figure(figsize=(10, 6))

# Curve WITHOUT data augmentation (shows overfitting)
plt.plot(epochs, hist_no_aug['val_loss'], 'r--', marker='x', label='Without Data Augmentation (Overfitting)')

# Curve WITH data augmentation (shows better generalization)
plt.plot(epochs, hist_aug['val_loss'], 'g-', marker='o', linewidth=2, label='With Data Augmentation (Our Model)')

plt.title('Impact of Data Augmentation on Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Save and show the plot
output_path = os.path.join(figures_dir, 'comparison_real.png')
plt.savefig(output_path)
print(f"✅ Plot generated: {output_path}")
plt.show()