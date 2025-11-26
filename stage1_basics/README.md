### **1️⃣ Imports and Setup**

```python
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
```

* `torch` → main PyTorch library ⚡
* `nn` → for building neural networks 🏗️
* `optim` → for optimization (like adjusting weights) 🛠️
* `DataLoader` → helps feed data into your model in batches 🍽️
* `datasets` & `transforms` → get datasets & preprocess images 🖼️
* `models` → prebuilt models like Vision Transformer (ViT) 🤖
* `tqdm` → makes a progress bar for loops ⏳

---

### **2️⃣ Device Selection**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

* Checks if you have a GPU (CUDA). If yes → use GPU ⚡, else CPU 🐢.
* GPU = much faster training for deep learning 💨

---

### **3️⃣ Image Transformations**

```python
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

* `Resize(224)` → ViT wants 224x224 images 📏
* `ToTensor()` → converts images to PyTorch-friendly format 🔢
* `Normalize()` → scales pixel values so the model learns better 🎚️

---

### **4️⃣ Load CIFAR-10**

```python
train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
```

* CIFAR-10 → 60k tiny images in 10 classes (cats, planes, etc.) 🐱✈️
* `DataLoader` → feeds batches of 32 images at a time 🔄
* `shuffle=True` → randomizes training data for better learning 🎲

---

### **5️⃣ Load Pretrained Vision Transformer**

```python
model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
num_features = model.heads.head.in_features
model.heads.head = nn.Linear(num_features, 10)
model = model.to(device)
```

* `vit_b_16` → Vision Transformer, a fancy image model 🖼️🤖
* We replace the last layer with `nn.Linear(..., 10)` because CIFAR-10 has 10 classes 🔄
* `.to(device)` → moves the model to GPU if available 🖥️

---

### **6️⃣ Loss & Optimizer**

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-5)
```

* `CrossEntropyLoss()` → standard for classification (e.g., cat vs dog) 🏷️
* `Adam` → smart way to update model weights 💪
* `lr=3e-5` → learning rate = how fast the model learns 🚀

---

### **7️⃣ Training Loop**

```python
epochs = 3
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
```

* Loop over epochs (full passes through dataset) 🔁
* `model.train()` → tells PyTorch we're training ⚙️
* Forward pass: `outputs = model(images)` → model guesses 🧠
* Compute loss: `loss = criterion(...)` → how wrong the guesses were ❌
* Backprop: `loss.backward()` → calculate gradients 🔄
* Update weights: `optimizer.step()` → model learns 📈
* `tqdm` shows progress bar ⏳

---

### **8️⃣ Evaluation**

```python
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
```

* `model.eval()` → turn off training stuff like dropout 📴
* `torch.no_grad()` → saves memory during evaluation 💾
* Compare predictions vs labels to get accuracy ✅

---

### **9️⃣ Save the Model**

```python
Path("models").mkdir(exist_ok=True, parents=True)
torch.save(model.state_dict(), "models/vit_cifar10.pth")
```

* Create a `models/` folder if it doesn't exist 📂
* Save model weights so you can load later 💾

---

### **Summary in One Line**

1. Load CIFAR-10 images 🖼️
2. Preprocess them (resize, normalize) 🎨
3. Load a pretrained Vision Transformer 🤖
4. Replace last layer for CIFAR-10 classes 🔄
5. Train for 3 epochs ⚡
6. Check accuracy ✅
7. Save the trained model 💾

<br>
