import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Root directory of the processed dataset.
            split (string): 'train' or 'val' split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Define class mapping: 0.0 for 'real', 1.0 for 'fake'
        self.class_map = {'real': 0.0, 'fake': 1.0}
        
        # Traverse subdirectories to collect image paths and labels
        if not os.path.exists(self.root_dir):
            print(f"Warning: Dataset directory not found: {self.root_dir}")
            return

        for label_name, label_val in self.class_map.items():
            class_path = os.path.join(self.root_dir, label_name)
                        
            if os.path.isdir(class_path):
                files = os.listdir(class_path)
                # We only consider .jpg files
                images = [f for f in files if f.lower().endswith('.jpg')]
                
                for img in images:
                    self.image_paths.append(os.path.join(class_path, img)) # Store full path
                    self.labels.append(label_val) # Store corresponding label (0.0 or 1.0)
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error reading image file {img_path}: {e}")
            # Return a dummy tensor on error to prevent crashing the training loop
            return torch.zeros((3, 224, 224)), torch.tensor(0.0) 

        # 2. Apply transformations (if any)
        if self.transform:
            image = self.transform(image)
        
        # 3. Return image and its label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label