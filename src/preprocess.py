import cv2
import os
import torch
import random
import shutil
from facenet_pytorch import MTCNN
from tqdm import tqdm


SOURCE_DIR = "data/raw"
DEST_DIR = "data/processed"
FRAMES_PER_VIDEO = 10    # We only extract 10 frames per video
MAX_VIDEOS_PER_CLASS = 100 # STOP after 100 videos per class to speed up testing
IMG_SIZE = 224
SPLIT_RATIO = 0.8        # 80% Train, 20% Val

def extract_faces_split_fast(source_dir, dest_dir):
    # 0. We clean the dest_dir if it exists
    if os.path.exists(dest_dir):
        print(f" We clean the folder {dest_dir}...")
        shutil.rmtree(dest_dir)

    # 1. Setup GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    mtcnn = MTCNN(
        image_size=IMG_SIZE, margin=0, keep_all=False, 
        select_largest=True, post_process=False, device=device
    )

    folders = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for folder_name in folders:
        folder_path = os.path.join(source_dir, folder_name)
        
        # We determine the label based on folder name
        label = 'fake' if 'synthesis' in folder_name.lower() or 'fake' in folder_name.lower() else 'real'
        
        # List every video in the folder
        all_videos = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        # Mixing the list for randomness
        random.seed(42)
        random.shuffle(all_videos)
        
        # Limiting the number of videos per class for faster testing
        if len(all_videos) > MAX_VIDEOS_PER_CLASS:
            print(f" We only keep {MAX_VIDEOS_PER_CLASS} videos for {label}")
            all_videos = all_videos[:MAX_VIDEOS_PER_CLASS]

        # The split index
        # It gives us the number of videos for training
        split_idx = int(len(all_videos) * SPLIT_RATIO)
        train_videos = all_videos[:split_idx]
        val_videos = all_videos[split_idx:]
        
        print(f" {label}: {len(train_videos)} Train / {len(val_videos)} Val")

        # Function to process a list of videos
        def process_list(video_list, split_name):
            output_folder = os.path.join(dest_dir, split_name, label)
            os.makedirs(output_folder, exist_ok=True)
            
            # We use tqdm for progress bar
            for video_file in tqdm(video_list, desc=f"Processing {split_name}/{label}"):
                video_path = os.path.join(folder_path, video_file)
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if total_frames > 0:
                    step = max(1, total_frames // FRAMES_PER_VIDEO)
                    frames_buffer = []
                    save_paths = []

                    for i in range(0, total_frames, step):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                        ret, frame = cap.read()
                        if not ret: break

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames_buffer.append(frame_rgb)
                        
                        base_name = video_file.split('.')[0]
                        save_name = f"{base_name}_frame{i}.jpg"
                        save_paths.append(os.path.join(output_folder, save_name))
                        
                        if len(frames_buffer) >= FRAMES_PER_VIDEO:
                            break
                    
                    if frames_buffer:
                        try:
                            mtcnn(frames_buffer, save_path=save_paths)
                        except: pass
                cap.release()

        process_list(train_videos, 'train')
        process_list(val_videos, 'val')

if __name__ == "__main__":
    extract_faces_split_fast(SOURCE_DIR, DEST_DIR)