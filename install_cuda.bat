@echo off
echo ========================================
echo AI Story Farm - CUDA Installation Script
echo ========================================
echo.
echo This script will install PyTorch with CUDA support
echo and then install all other dependencies.
echo.
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: nvidia-smi not found. Make sure NVIDIA drivers are installed.
    echo.
) else (
    echo NVIDIA GPU detected!
    nvidia-smi --query-gpu=name --format=csv,noheader
    echo.
)

echo Installing PyTorch with CUDA 12.1...
echo (If you need CUDA 11.8, edit this script and change cu121 to cu118)
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if %errorlevel% neq 0 (
    echo.
    echo ERROR: PyTorch CUDA installation failed!
    echo Trying CUDA 11.8 instead...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)

echo.
echo Verifying CUDA installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

if %errorlevel% neq 0 (
    echo ERROR: Python verification failed!
    pause
    exit /b 1
)

echo.
echo Installing other dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo Installation complete!
echo ========================================
pause

