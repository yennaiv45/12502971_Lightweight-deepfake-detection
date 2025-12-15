import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import numpy as np
import os
import time

# --- IMPORTS DE NOS MODULES ---
from src.dataset import DeepfakeDataset
from src.model import DeepFakeMobileNet

# --- CONFIGURATION (HYPERPARAMÈTRES) ---
BATCH_SIZE = 32          # Baisser à 16 si "Out of Memory"
LEARNING_RATE = 1e-4     # Petit LR pour ne pas casser les poids pré-entraînés
EPOCHS = 5               # 5 époques suffisent pour voir si ça apprend (Baseline)
NUM_WORKERS = 0          # Mettre 0 pour éviter les bugs sous Windows, sinon 2 ou 4 sous Linux
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Si Mac M1/M2
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

print(f"🚀 Entraînement lancé sur : {DEVICE}")

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    loop = tqdm(loader, desc="Training", leave=False)
    
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
        
        optimizer.zero_grad()
        
        # --- MIXED PRECISION (Accélération GPU) ---
        # Si GPU NVIDIA, on utilise autocast
        if DEVICE.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # CPU ou MPS (Mac) standard
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        
        # Stockage pour métriques (sigmoid pour passer de logits à probabilité 0-1)
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(probs)
        
        loop.set_postfix(loss=loss.item())
        
    avg_loss = running_loss / len(loader)
    
    # Calcul métriques (gestion cas où une seule classe est présente dans le batch)
    try:
        epoch_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        epoch_auc = 0.5 # Fallback si pas assez de données variées
        
    epoch_acc = accuracy_score(all_labels, np.round(all_preds))
    
    return avg_loss, epoch_acc, epoch_auc

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probs)
            
    avg_loss = running_loss / len(loader)
    try:
        val_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        val_auc = 0.5
    val_acc = accuracy_score(all_labels, np.round(all_preds))
    
    return avg_loss, val_acc, val_auc

def main():
    # 1. Transformations (Standard ImageNet)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # Data Augmentation simple
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Datasets & Loaders
    print("📁 Chargement des données...")
    # NOTE: Pour l'Assignment 2, on utilise 'train' pour l'entrainement
    # Si tu n'as pas créé de dossier 'val', on utilise 'train' aussi pour valider (juste pour tester le code)
    # Dans un vrai projet, il faut séparer les dossiers.
    train_dataset = DeepfakeDataset(root_dir="data/processed", split='train', transform=train_transform)
    
    # HACK : Si pas de dossier val, on split manuellement ou on utilise train (déconseillé mais ok pour debug)
    val_dir = "data/processed/val" if os.path.exists("data/processed/val") else "data/processed/train"
    val_dataset = DeepfakeDataset(root_dir="data/processed", split='train', transform=val_transform) 

    if len(train_dataset) == 0:
        print("❌ Erreur : Dataset vide ! Lance preprocess.py d'abord.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    print(f"📊 Images d'entraînement : {len(train_dataset)}")

    # 3. Modèle
    model = DeepFakeMobileNet(pretrained=True).to(DEVICE)
    
    # 4. Loss & Optimizer
    # pos_weight > 1 pour donner plus d'importance aux Fakes s'ils sont rares (ou inversement)
    # Pour la baseline, on laisse à 1.0 (équilibré) ou on ajuste selon le dataset.
    pos_weight = torch.tensor([1.0]).to(DEVICE) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

    # 5. Boucle
    best_auc = 0.0
    start_time = time.time()
    
    print("🔥 Démarrage de l'entraînement...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion)
        
        print(f"   Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f}")
        print(f"   Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")
        
        # Sauvegarde du meilleur modèle
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_model.pth")
            print("   ✅ Modèle sauvegardé (Nouveau record AUC)")

    total_time = time.time() - start_time
    print(f"\n🏁 Entraînement terminé en {total_time/60:.1f} minutes.")
    print(f"🏆 Meilleur AUC atteint : {best_auc:.4f}")

if __name__ == "__main__":
    main()