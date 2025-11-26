## Stage 3 — "Raj Classification Transformer Thingy"

Goal: Build your first real, domain-specific transformer-based classifier.

Tasks:

1. **Clean + format Raj's dataset for image classification**

   * Create a PyTorch dataset/dataloader
   * Normalize patches
   * Ensure class labels match JSON/metadata

2. **Build a Vision Transformer for his classes**

   * Fine-tune ViT on tile data
   * Log metrics (TensorBoard or `wandb`)
   * Compare with your CNN baseline

3. **Experiment time**

   * Add augmentation
   * Mixup / Cutmix
   * Try an attention model vs your ResNet
   * Compare accuracy + F1 + confusion matrix

By the end of this stage:  
You actually *get* transformers.  
Like not just "I can run PyTorch," but "I understand why attention makes this better."

<br>
