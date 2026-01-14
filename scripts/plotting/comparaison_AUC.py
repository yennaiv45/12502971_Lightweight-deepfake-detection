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

epochs = range(1, len(hist_aug['val_auc']) + 1)

# --- Plot AUC Comparison ---
plt.figure(figsize=(10, 6))

# Curve WITHOUT data augmentation
plt.plot(epochs, hist_no_aug['val_auc'], 'r--', marker='x', label='Without Data Augmentation')

# Curve WITH data augmentation
plt.plot(epochs, hist_aug['val_auc'], 'g-', marker='o', linewidth=2, label='With Data Augmentation (Our Model)')

plt.title('Impact of Data Augmentation on Performance (AUC)')
plt.xlabel('Epochs')
plt.ylabel('Validation AUC')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)

# Save and show the plot
output_path = os.path.join(figures_dir, 'comparison_auc.png')
plt.savefig(output_path)
print(f"✅ Plot generated: {output_path}")
plt.show()
