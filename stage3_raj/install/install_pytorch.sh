#!/bin/bash
# Installation script for PyTorch with GPU or CPU support

set -e

echo "=========================================="
echo "PyTorch Installation Script"
echo "=========================================="
echo ""

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""

    # Try to detect CUDA version
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        echo "CUDA Version: $CUDA_VERSION"
    else
        echo "nvcc not found - checking nvidia-smi for CUDA version"
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        echo "CUDA Version (from driver): $CUDA_VERSION"
    fi
    echo ""

    # Determine CUDA wheel version
    if [[ $CUDA_VERSION == 12.4* ]] || [[ $CUDA_VERSION == 12.5* ]]; then
        TORCH_INDEX="cu124"
        echo "Using PyTorch with CUDA 12.4 support"
    elif [[ $CUDA_VERSION == 12.1* ]] || [[ $CUDA_VERSION == 12.2* ]] || [[ $CUDA_VERSION == 12.3* ]]; then
        TORCH_INDEX="cu121"
        echo "Using PyTorch with CUDA 12.1 support"
    elif [[ $CUDA_VERSION == 11.8* ]]; then
        TORCH_INDEX="cu118"
        echo "Using PyTorch with CUDA 11.8 support"
    else
        echo "⚠ Unsupported CUDA version: $CUDA_VERSION"
        echo "Defaulting to CUDA 12.1"
        TORCH_INDEX="cu121"
    fi

    INSTALL_CMD="pip install torch torchvision --index-url https://download.pytorch.org/whl/$TORCH_INDEX"
else
    echo "⚠ No NVIDIA GPU detected"
    echo "Installing CPU-only version"
    TORCH_INDEX="cpu"
    INSTALL_CMD="pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
fi

echo ""
echo "Installation command:"
echo "  $INSTALL_CMD"
echo ""

# Ask for confirmation
read -p "Proceed with installation? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing PyTorch..."
    eval $INSTALL_CMD

    echo ""
    echo "Installing other dependencies..."
    pip install pillow numpy tqdm scikit-learn

    echo ""
    echo "=========================================="
    echo "Installation complete!"
    echo "=========================================="
    echo ""
    echo "Testing PyTorch installation..."
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

    if [ "$TORCH_INDEX" != "cpu" ]; then
        python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
    fi
else
    echo "Installation cancelled"
    exit 1
fi
