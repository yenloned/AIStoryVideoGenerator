#!/bin/bash

echo "========================================"
echo "AI Story Farm - CUDA Installation Script"
echo "========================================"
echo ""
echo "This script will install PyTorch with CUDA support"
echo "and then install all other dependencies."
echo ""

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected!"
    nvidia-smi --query-gpu=name --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. Make sure NVIDIA drivers are installed."
    echo ""
fi

echo "Installing PyTorch with CUDA 12.1..."
echo "(If you need CUDA 11.8, edit this script and change cu121 to cu118)"
echo ""

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: PyTorch CUDA installation failed!"
    echo "Trying CUDA 11.8 instead..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

echo ""
echo "Verifying CUDA installation..."
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

if [ $? -ne 0 ]; then
    echo "ERROR: Python verification failed!"
    exit 1
fi

echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"

