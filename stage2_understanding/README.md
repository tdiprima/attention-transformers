## Stage 2 — "Okay bet, now let's understand what's happening."

Goal: Peel the transformer open a bit without going full PhD.

Tasks:

1. **Write a tiny "manual vision transformer" module:**

   * Patchify an image manually
   * Add positional embeddings
   * Run it through a tiny `nn.TransformerEncoder`
   * Do classification

2. **Generate and visualize attention maps**

   * Grab attention from the model
   * Display which parts of the image it focuses on

3. **Swap between architectures**

   * ViT-base
   * ViT-small
   * Data-efficient ViT
   * Maybe DeiT (Distilled Vision Transformer)

You'll learn:

* Patch embeddings
* Positional embeddings
* Self-attention shapes
* Multi-head attention
* Why transformers get the whole-image context instantly

This is the "oooohhh THAT'S how they think" stage.

<br>
