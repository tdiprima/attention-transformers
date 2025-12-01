# Installation Guide

This guide covers installing PyTorch with GPU or CPU support for the Raj ViT training project.

## Quick Start (Automated)

### Using the installation script (recommended):

```bash
chmod +x install_pytorch.sh
./install_pytorch.sh
```

This script will:

- Auto-detect if you have an NVIDIA GPU
- Detect your CUDA version
- Install the correct PyTorch version
- Install all other dependencies

## Manual Installation

### 1. Check Your CUDA Version

If you have an NVIDIA GPU:

```bash
nvidia-smi
```

Look for "CUDA Version" in the output (e.g., 12.1, 11.8).

### 2. Install PyTorch with GPU Support

Choose the command based on your CUDA version:

**CUDA 12.4+:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**CUDA 12.1-12.3:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CPU only (no GPU):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Other Dependencies

```bash
pip install pillow numpy tqdm scikit-learn
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

If CUDA is available, check GPU details:

```bash
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Using uv (alternative)

If you're using `uv` instead of `pip`:

**GPU (CUDA 12.1):**

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install pillow numpy tqdm scikit-learn
```

**CPU:**

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install pillow numpy tqdm scikit-learn
```

## Troubleshooting

### "CUDA out of memory" error

- Reduce batch size: `--batch_size 8` or `--batch_size 4`
- Reduce number of workers: `--num_workers 2`

### PyTorch not detecting GPU
1. Check NVIDIA driver: `nvidia-smi`
2. Verify CUDA installation: `nvcc --version`
3. Reinstall PyTorch with correct CUDA version (see above)
4. Check compatibility: https://pytorch.org/get-started/locally/

### Slow training on GPU
- Make sure you're using GPU version of PyTorch (check with verification command above)
- Monitor GPU usage: `nvidia-smi -l 1` (updates every second)
- Ensure data is being loaded to GPU (script does this automatically)

## Files in This Directory

- `pyproject.toml` - Main project configuration (GPU/CPU agnostic)
- `pyproject.cpu.toml` - CPU-specific configuration
- `install_pytorch.sh` - Automated installation script
- `INSTALL.md` - This file

## Training After Installation

Once installed, you can start training:

```bash
# Check dataset
python -c "from raj_dataset import RajDataset; ds = RajDataset('/data/erich/raj/data/train'); print(f'Images: {len(ds)}, Classes: {ds.num_classes()}')"

# Start training (uses GPU automatically if available)
python train_raj_vit.py --root_dir /data/erich/raj/data/train --epochs 10 --batch_size 32

# Or with uv
uv run train_raj_vit.py --root_dir /data/erich/raj/data/train --epochs 10 --batch_size 32
```

The training script automatically uses GPU if available, falls back to CPU if not.

<br>
