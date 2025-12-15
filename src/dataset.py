import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Répertoire racine (ex: 'data/processed')
            split (string): 'train' ou 'val'
            transform (callable, optional): Transformations à appliquer (augmentation, resize)
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Définition des classes : 0 = Real, 1 = Fake
        # Assurez-vous que vos dossiers s'appellent 'real' et 'fake' (minuscule ou majuscule géré)
        self.class_map = {'real': 0.0, 'fake': 1.0}
        
        # Parcours des dossiers
        if not os.path.exists(self.root_dir):
            print(f"⚠️ Attention : Le dossier {self.root_dir} n'existe pas encore.")
            return

        for label_name, label_val in self.class_map.items():
            class_path = os.path.join(self.root_dir, label_name)
                        
            if os.path.isdir(class_path):
                files = os.listdir(class_path)
                # On ne garde que les images
                images = [f for f in files if f.lower().endswith(('.jpg'))]
                
                for img in images:
                    self.image_paths.append(os.path.join(class_path, img))
                    self.labels.append(label_val)
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Charger l'image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Erreur de lecture : {img_path}")
            return torch.zeros((3, 224, 224)), torch.tensor(0.0) # Retourne un dummy en cas d'erreur

        # 2. Appliquer les transformations (Resize, Normalize)
        if self.transform:
            image = self.transform(image)
        
        # 3. Retourner Image + Label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label