# 1. Image de base
FROM python:3.9-slim

# 2. Dossier de travail
WORKDIR /app

# 3. Installation des dépendances système
# IMPORTANT : OpenCV nécessite libgl1-mesa-glx pour fonctionner dans Docker
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Installation de PyTorch (Version CPU Légère)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. Copie et installation des autres dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copie du code source UNIQUEMENT
# (Le modèle sera téléchargé par le script Python au lancement)
COPY src/ src/
COPY PROJECT_DOCUMENTATION.md . 
# J'ai ajouté le MD au cas où, mais tu peux l'enlever

# 7. Configuration
ENV PYTHONPATH="/app"
EXPOSE 8501

# 8. Lancement
# Assure-toi que le chemin vers ton app est bon. 
# Si app_streamlit.py est dans src/, c'est bon.
CMD ["streamlit", "run", "src/app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]