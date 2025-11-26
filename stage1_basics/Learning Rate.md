## Learning Rate

* `3e-5` = 0.00003, which is **super small**.
* For **fine-tuning a pretrained transformer** like ViT, this is pretty typical. Why? 🤔

  * Transformers already learned a lot from ImageNet pretraining.
  * A **small LR** prevents "forgetting" that knowledge while still letting it adapt to CIFAR-10.

If you make it bigger, like `1e-3` (0.001), your model might:

* Learn faster 🏎️
* But **risk messing up pretrained weights**, leading to worse accuracy ⚠️

💡 Rule of thumb:

* Pretrained transformers → `1e-5` to `5e-5` for fine-tuning ✅
* Training from scratch → usually `1e-3` or so

Since you're using **ViT pretrained**, `3e-5` is totally fine.

---

Bet 😎. Here's a quick way to **check if your learning rate is in the sweet spot** while training:

---

### **1️⃣ Watch the Loss**

* During training, your code already prints loss per batch/epoch.
* **Good LR:** loss gradually decreases 📉
* **Too high:** loss jumps around or explodes 🔥
* **Too low:** loss barely moves, training is super slow 🐢

---

### **2️⃣ Use a Learning Rate Finder**

PyTorch has a trick called **LR range test**:

* Start with a super tiny LR (`1e-7`)
* Gradually increase each batch up to something big (`1e-1`)
* Plot loss vs LR
* The **optimal LR** is usually **just before loss starts blowing up** 💣

Tools:

* `torch_lr_finder` package
* Or manually: increase LR exponentially for a few batches and plot

---

### **3️⃣ Watch Gradients (Advanced)**

* If gradients **explode** → LR too high ⚠️
* If gradients are **tiny / zero** → LR too low 💤

---

💡 **Quick practical tip for your ViT on CIFAR-10**:

* With `3e-5`, loss should slowly decrease over 3 epochs.
* If after 1 epoch loss barely moves, you can nudge it to `5e-5`.
* Don't go above `1e-4` unless you want a fiery crash 🔥

---

Bet 😎. Here's a **simple tweak** to your training loop that tracks the **average loss per batch** and warns you if the learning rate might be too high or too low.

You basically just monitor the loss trend as you train:

```python
prev_loss = None

for epoch in range(epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")

    # --- LR check ---
    if prev_loss is not None:
        if avg_loss > prev_loss * 1.05:  # loss went up >5%
            print("⚠️ Warning: Loss increased! LR might be too high.")
        elif avg_loss < prev_loss * 0.95:  # loss decreased <5%
            print("ℹ️ Loss barely changed. LR might be too low.")
    prev_loss = avg_loss
```

### **How it works:**

* Compares **current epoch avg loss** vs previous epoch ✅
* If loss **jumped up** → probably too high LR 🔥
* If loss **barely changed** → probably too low 🐢
* Gives you a gentle **console warning** without stopping training ⚡

---

Bet 😎. Here's a **super beginner-friendly way to plot your loss per batch in real time** using `matplotlib`. This lets you **see if your LR is too high or low** while training.

---

### **1️⃣ Add imports**

At the top of your script:

```python
import matplotlib.pyplot as plt
from IPython.display import clear_output
```

> `clear_output()` is just to **update the plot in place** instead of making a million new figures.

---

### **2️⃣ Modify your training loop**

Inside your loop, track losses and plot them:

```python
batch_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)

    for i, (images, labels) in enumerate(loop):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_losses.append(loss.item())

        # Update progress bar
        loop.set_postfix(loss=loss.item())

        # --- Real-time plot ---
        if i % 10 == 0:  # update plot every 10 batches
            clear_output(wait=True)
            plt.figure(figsize=(8,4))
            plt.plot(batch_losses, label="Batch Loss")
            plt.xlabel("Batch")
            plt.ylabel("Loss")
            plt.title("Training Loss per Batch")
            plt.legend()
            plt.show()
```

---

### **3️⃣ What this does**

* Plots **loss after every 10 batches** 📊
* If the curve:

  * **Jumps up** → LR too high 🔥
  * **Barely drops** → LR too low 🐢
  * **Smoothly goes down** → perfect ✅

---

💡 **Pro tip:**
You can combine this with the **epoch-level warnings** I gave earlier for an extra safety net.

<br>
