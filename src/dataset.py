import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Root repo 
            split (string): 'train' or 'val'
            transform (callable, optional): Optional transform to be applied
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Class definition 0 = Real, 1 = Fake
        self.class_map = {'real': 0.0, 'fake': 1.0}
        
        # Go through folders and collect image paths and labels
        if not os.path.exists(self.root_dir):
            print(f"The folder {self.root_dir} does not exist.")
            return

        for label_name, label_val in self.class_map.items():
            class_path = os.path.join(self.root_dir, label_name)
                        
            if os.path.isdir(class_path):
                files = os.listdir(class_path)
                # We only consider .jpg files
                images = [f for f in files if f.lower().endswith(('.jpg'))]
                
                for img in images:
                    self.image_paths.append(os.path.join(class_path, img)) # Full path
                    self.labels.append(label_val) # Corresponding label (0.0 or 1.0)
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Erreur de lecture : {img_path}")
            return torch.zeros((3, 224, 224)), torch.tensor(0.0) # Return dummy data if error 

        # 2. Apply transformations for data augmentation 
        if self.transform:
            image = self.transform(image)
        
        # 3. Return image and label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label