# Base Image: NVIDIA PyTorch 24.10 (Python 3.10, CUDA 12.6, PyTorch 2.5)
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Définition du répertoire de travail
WORKDIR /workspace

# ---------------------------------------------------------------------------
# 1. INSTALLATION SYSTÈME & NETTOYAGE
# ---------------------------------------------------------------------------
# On installe les libs graphiques requises par OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. GESTION FINE DE OPENCV & TIMM
# ---------------------------------------------------------------------------
# On désinstalle les versions pré-fournies qui posent conflit
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python timm \
    && rm -rf /usr/local/lib/python3.10/dist-packages/cv2 \
    && rm -rf /usr/local/lib/python3.10/site-packages/cv2

# Installation propre :
# - OpenCV Headless (pour éviter les erreurs GUI sur serveur)
# - Timm (version récente, car nous avons patché le code pour le supporter)
RUN pip install --no-cache-dir \
    opencv-python-headless==4.8.0.74 \
    timm

# ---------------------------------------------------------------------------
# 3. INSTALLATION MAMBA (VIA TES WHEELS)
# ---------------------------------------------------------------------------
# L'ordre est important : causal-conv1d avant mamba-ssm (ou ensemble)
RUN pip install --no-cache-dir \
    https://github.com/Dorianhgn/OpenSTL/releases/download/v1.0-wheels/causal_conv1d-1.4.0-cp310-cp310-linux_x86_64.whl \
    https://github.com/Dorianhgn/OpenSTL/releases/download/v1.0-wheels/mamba_ssm-2.2.2-cp310-cp310-linux_x86_64.whl

# ---------------------------------------------------------------------------
# 4. DÉPENDANCES PYTHON (JVM & RUNTIME)
# ---------------------------------------------------------------------------
COPY requirements/runtime.txt /tmp/runtime.txt
COPY requirements/jvm.txt /tmp/jvm.txt

RUN pip install --no-cache-dir -r /tmp/runtime.txt && \
    pip install --no-cache-dir -r /tmp/jvm.txt

# ---------------------------------------------------------------------------
# 5. CONFIGURATION FINALE
# ---------------------------------------------------------------------------
ENV PYTHONPATH=/workspace:${PYTHONPATH}

# Commande par défaut
CMD ["/bin/bash"]