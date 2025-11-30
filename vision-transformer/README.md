# 🪄 Transformers

## 🚀 So... what even *is* a Transformer?

Think of a transformer as a model with goldfish-level attention span *but* the goldfish is actually a genius.  
It looks at **everything** in your input at once and decides what matters most.

Originally made for language, but now it's basically the main character of machine learning — images, text, audio, proteins, vibes, everything.

The magic sauce?  
👉 **Self-attention** — the model literally "checks out all the parts" of whatever you feed it and picks the parts that matter.

---

## 🧠 Core Ideas (minus the headache)

### 1. ✨ Self-Attention

Imagine you're reading a sentence and your brain goes "wait, that word connects to that other word."  
That's literally what self-attention does.

Each piece of the input makes three versions of itself:

* **Query:** "what am I looking for?"
* **Key:** "what do I have?"
* **Value:** "what am I actually saying?"

Then the model matches Q with K, decides relevance, and grabs the V's it needs.

### 2. 👑 Multi-Head Attention

Instead of doing that once, it does it *multiple times at the same time*.  
Like having multiple little brains, each checking out different relationships.

### 3. 📍 Positional Embeddings

Transformers don't naturally understand order.  
So we give them "this came first, this came second" stickers.

### 4. 🔧 Feed-Forward Layers

After attention, each token gets passed through a small MLP for extra glow-up.  

### 5. 🔁 Residuals + LayerNorm

These keep the model from drifting into chaos and exploding your loss.  
(It's giving "mental health support for neural nets.")

---

# 👁️ Vision Transformers (ViT) — the image edition

Transformers don't take raw images, so we do some surgery:

1. **Split the image into patches** (like chopping an image into lil' squares)
2. **Flatten each patch**
3. **Turn each into an embedding vector**
4. **Add a special token** (the "CLS" token — the model's group project leader)
5. **Add position info**
6. **Send it through a bunch of transformer blocks**
7. **Use the CLS token's final vibes to classify**

That's it. It's like treating the image as a bunch of "words".

---

# 💪 Why Transformers Are the Main Character

* They look at *everything* at once → huge brain energy
* They handle long-range dependencies like a champ
  (CNNs can't see far unless stacked deep af)
* Scale like crazy — give them more data and they just get better
* Easy to visualize what they're "paying attention" to

---

# 🥊 Transformers vs CNNs (quick vibe check)

| Thing                | CNNs       | Transformers            |
| -------------------- | ---------- | ----------------------- |
| Built-in assumptions | Lots       | Basically none          |
| See whole image?     | Not really | Yes, immediately        |
| Needs lots of data?  | No         | Oh absolutely           |
| Compute cost         | Chill      | Girly is *expensive*    |
| Performance          | Great      | God-tier w/ enough data |

---

# 🔧 Hyperparams but make it simple

* **embed_dim** = size of each token's brain
* **depth** = number of transformer layers (model's maturity level)
* **num_heads** = how many attention "brains" at once
* **mlp_ratio** = how thicc the MLP is
* **patch_size** = size of image chunks
* **dropout** = how often we make the model "forget" things to avoid overfitting

If you don't have a ton of data → smaller embed_dim + fewer layers is safer.

---

# 🎨 Training Vibes

* Use **AdamW** — it's the standard "I lift" optimizer
* LR around **1e-4**, warmup + cosine decay
* **Data augmentation** is crucial for ViT (mixup, flips, color jitter)
* Regularization is your friend (dropout, label smoothing)

---

# 📦 What they're used for besides classification

Transformers run like 75% of modern ML:

* Object detection
* Segmentation
* Image generation (DALL-E, Diffusion stuff)
* Medical imaging
* Multi-modal models (CLIP, BLIP)
* Text → everything (LLMs, translation, summarization, chatbots)

<br>
