import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
import requests
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
# Assure-toi que src.model est bien accessible via PYTHONPATH ou imports relatifs
# Si "from model" ne marche pas dans Docker, essaie "from src.model"
try:
    from model import DeepFakeMobileNet
except ImportError:
    from src.model import DeepFakeMobileNet

# --- CONFIGURATION ---
# 👇 REMPLACE CECI PAR LE LIEN QUE TU AS COPIÉ DEPUIS TES RELEASES GITHUB
MODEL_URL = "https://github.com/yennaiv45/12502971_Lightweight-deepfake-detection/releases/download/v1.0/best_model.pth"
MODEL_FILENAME = "best_model.pth"

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🛡️",
    layout="centered"
)

# --- DEVICE & MODEL SETUP ---
DEVICE = torch.device('cpu') 

def download_model_if_missing():
    """Télécharge le modèle depuis GitHub si le fichier n'existe pas localement."""
    if not os.path.exists(MODEL_FILENAME):
        with st.spinner(f"📥 Téléchargement du modèle en cours depuis GitHub... (Une seule fois)"):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status() # Vérifie que le lien fonctionne (pas de 404)
                with open(MODEL_FILENAME, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("✅ Modèle téléchargé avec succès !")
            except Exception as e:
                st.error(f"❌ Erreur critique : Impossible de télécharger le modèle.\n{e}")
                st.stop()

@st.cache_resource
def load_model_and_tools():
    """
    Loads the MTCNN face detector, the pre-trained DeepFakeMobileNet model, and image transforms.
    The @st.cache_resource decorator ensures this function runs only once.
    """
    # 1. Check and download model FIRST
    download_model_if_missing()

    print("⏳ Loading model and tools...")
    
    # 2. Load MTCNN for face detection
    mtcnn = MTCNN(keep_all=False, select_largest=True, device=DEVICE)
    
    # 3. Load the trained model
    model = DeepFakeMobileNet(pretrained=False)
    try:
        model.load_state_dict(torch.load(MODEL_FILENAME, map_location=DEVICE))
    except FileNotFoundError:
        st.error(f"❌ Error: The model file '{MODEL_FILENAME}' was not found despite download attempt.")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model weights: {e}")
        return None, None, None
        
    model.to(DEVICE)
    model.eval()
    
    # 4. Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return model, mtcnn, transform

model, mtcnn, val_transform = load_model_and_tools()

# --- PREDICTION LOGIC ---
def process_video(video_path):
    """
    Processes a video file to detect deepfakes. It extracts faces from a subset of frames,
    runs them through the model, and returns the average prediction score.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return None, "Error reading the video file."

    # Analyze only 10 evenly spaced frames for faster processing
    num_frames_to_check = 10
    step = max(1, total_frames // num_frames_to_check)
    
    predictions = []
    frames_processed = 0
    
    progress_bar = st.progress(0, text="Analyzing video...")
    
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
        
        # Convert from BGR (OpenCV default) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Detect and crop faces
        try:
            boxes, _ = mtcnn.detect(pil_img)
            
            if boxes is not None:
                for box in boxes:
                    # Manually crop the detected face
                    face = pil_img.crop(box)
                    
                    # Prepare the face tensor for the model
                    tensor = val_transform(face).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        output = model(tensor)
                        prob = torch.sigmoid(output).item()
                        predictions.append(prob)
        except Exception:
            # Silently pass if no face is detected in a frame
            pass
        
        frames_processed += 1
        progress = min(frames_processed / num_frames_to_check, 1.0)
        progress_bar.progress(progress)
            
    cap.release()
    progress_bar.empty() # Remove the progress bar when done
    
    if not predictions:
        return None, "No faces were detected in the video."

    return np.mean(predictions), "Analysis successful."

# --- USER INTERFACE (UI) ---
st.title("🛡️ Lightweight Deepfake Detection")
st.markdown("""
This app uses **MobileNetV3** to analyze videos.
It automatically extracts faces and calculates a manipulation probability.
""")

uploaded_file = st.file_uploader("Choose a video file (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])

if uploaded_file is not None and model is not None:
    # 1. Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name

    # 2. Display the uploaded video
    st.video(uploaded_file)
    
    if st.button("Launch Analysis 🚀"):
        with st.spinner('Analyzing frames... (This may take a moment)'):
            # Pass the path of the temporary file to the processing function
            score, message = process_video(temp_video_path)
        
        # 3. Clean up the temporary file
        os.remove(temp_video_path)
        
        # 4. Display the result
        if score is None:
            st.warning(f"Could not complete analysis: {message}")
        else:
            if score > 0.5:
                st.error(f"🚨 **DEEPFAKE DETECTED**")
                st.metric(label="Fake Probability", value=f"{score*100:.2f}%")
            else:
                st.success(f"✅ **LIKELY REAL**")
                st.metric(label="Real Confidence", value=f"{(1-score)*100:.2f}%")
elif model is None:
    st.error("Model failed to load. Please check your internet connection or the GitHub URL.")