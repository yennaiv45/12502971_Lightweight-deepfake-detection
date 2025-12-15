import cv2
import os
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

# Configuration
SOURCE_DIR = "data/raw"  # Adapte selon ton dossier
DEST_DIR = "data/processed"
FRAMES_PER_VIDEO = 40  # Cible suggérée par l'article LightFakeDetect 
IMG_SIZE = 224

def extract_faces(source_dir, dest_dir):
    # Initialiser MTCNN (sur GPU si dispo pour aller 10x plus vite)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"------------------------------------------------")
    print(f"🚀 LE SCRIPT TOURNE SUR : {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"------------------------------------------------")

    # 2. Passer le device à MTCNN
    mtcnn = MTCNN(
        image_size=224, 
        margin=0, 
        keep_all=False, 
        select_largest=True, 
        post_process=False, 
        device=device  # <--- TRÈS IMPORTANT
        )
    # Parcourir les dossiers 'Celeb-real', 'Celeb-synthesis', 'YouTube-real'
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Déterminer le label (0: Real, 1: Fake)
        # Adapter selon les noms de dossiers de ton dataset
        label = 'fake' if 'synthesis' in folder_name.lower() else 'real'
        output_path = os.path.join(dest_dir, 'train', label) # On met tout dans train pour l'instant
        os.makedirs(output_path, exist_ok=True)

        videos = [f for f in os.listdir(folder_path) if f.endswith(('.mp4'))]
        
        print(f"Processing folder: {folder_name} ({len(videos)} videos) -> Label: {label}")

        for video_file in tqdm(videos):
            video_path = os.path.join(folder_path, video_file)
            cap = cv2.VideoCapture(video_path)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0: continue
            
            # Calculer le saut (step) pour avoir environ 40-50 frames uniformément réparties
            step = max(1, total_frames // FRAMES_PER_VIDEO)
            
            count_saved = 0
            for frame_idx in range(0, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret: break

                # OpenCV est en BGR, MTCNN a besoin de RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Détection et Sauvegarde
                # save_path force MTCNN à sauvegarder directement l'image
                base_name = video_file.split('.')[0]
                save_name = f"{base_name}_frame{frame_idx}.jpg"
                save_file_path = os.path.join(output_path, save_name)
                
                try:
                    # mtcnn retourne le tenseur ou None, mais avec save_path il sauve sur disque
                    mtcnn(frame_rgb, save_path=save_file_path)
                    count_saved += 1
                except Exception as e:
                    # Parfois la détection échoue (image floue, profil extrême)
                    pass

            cap.release()

if __name__ == "__main__":
    if not os.path.exists(SOURCE_DIR):
        print(f"Erreur: Le dossier {SOURCE_DIR} n'existe pas. Vérifie ton chemin.")
    else:
        extract_faces(SOURCE_DIR, DEST_DIR)
        print("Prétraitement terminé !")