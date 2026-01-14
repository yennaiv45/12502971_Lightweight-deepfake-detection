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
import json  # <--- IMPORTANT: For saving the history

# --- LOCAL MODULE IMPORTS ---
from src.dataset import DeepfakeDataset
from src.model import DeepFakeMobileNet

# --- CONFIGURATION (HYPERPARAMETERS) ---
BATCH_SIZE = 32          
LEARNING_RATE = 1e-4     
EPOCHS = 10              
NUM_WORKERS = 0          # 0 for Windows compatibility
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# If on Apple Silicon (M1/M2)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

print(f"🚀 Starting training on: {DEVICE}")

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    loop = tqdm(loader, desc="Training", leave=False)
    
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
        
        optimizer.zero_grad()
        
        # --- MIXED PRECISION (FOR GPU ACCELERATION) ---
        if DEVICE.type == 'cuda':
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(probs)
        
        loop.set_postfix(loss=loss.item())
        
    avg_loss = running_loss / len(loader)
    
    try:
        epoch_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        epoch_auc = 0.5 
        
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

# === EXPERIMENT SETUP ===
USE_AUGMENTATION = False  # <--- Set to True for the first run, then False for the second
# =====================================

def main():
    print(f"🔬 Starting experiment run: Data Augmentation = {USE_AUGMENTATION}")

    # 1. Define transforms based on the experiment
    if USE_AUGMENTATION:
        # This is your robust pipeline
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        save_name = "history_with_aug.json" # Specific filename
    else:
        # Naive pipeline (this will likely overfit)
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # No random transformations here, just formatting.
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        save_name = "history_no_aug.json" # Specific filename

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Datasets & Loaders
    print("📁 Loading data...")
    train_dataset = DeepfakeDataset(root_dir="data/processed", split='train', transform=train_transform)
    val_dataset = DeepfakeDataset(root_dir="data/processed", split='val', transform=val_transform) 

    if len(train_dataset) == 0:
        print("❌ Error: Training dataset is empty! Run preprocess.py first.")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    print(f"📊 Train images: {len(train_dataset)} | Validation images: {len(val_dataset)}")

    # 3. Model
    model = DeepFakeMobileNet(pretrained=True).to(DEVICE)
    
    # 4. Loss & Optimizer
    pos_weight = torch.tensor([1.0]).to(DEVICE) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

    # --- INITIALIZE HISTORY ---
    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': []
    }

    # 5. Training Loop
    best_auc = 0.0
    start_time = time.time()
    
    print("🔥 Starting training loop...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        # Validate
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion)
        
        # --- SAVE METRICS TO HISTORY ---
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['train_auc'].append(float(train_auc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        history['val_auc'].append(float(val_auc))

        print(f"   Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f}")
        print(f"   Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")
        
        # Save the best performing model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_model.pth")
            print("   ✅ New best model saved (based on AUC)")

    total_time = time.time() - start_time
    print(f"\n🏁 Training complete in {total_time/60:.1f} minutes.")
    print(f"🏆 Best validation AUC: {best_auc:.4f}")

    # --- SAVE HISTORY TO FILE ---
    print(f"💾 Saving training history to '{save_name}'...")
    with open(save_name, 'w') as f:
        json.dump(history, f)

if __name__ == "__main__":
    main()
