# raj_dataset.py
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RajDataset(Dataset):
    """
    RajDataset supports two modes:
      1) CSV annotations: annotations_file CSV must have 'image_path' and 'label' columns
      2) Folder layout: root_dir contains class subfolders (e.g., root/classA/*.png)
    Returns: (image_tensor, label_index, image_path)
    """

    def __init__(
        self,
        root_dir=None,
        annotations_file=None,
        transform=None,
        img_size=224,
        ensure_exists=True,
    ):
        """
        Args:
            root_dir (str): root folder with subfolders per class (optional if using CSV)
            annotations_file (str): path to CSV with columns ['image_path','label'] (can be absolute or relative)
            transform (torchvision.transforms): optional transform; if None a default transform is used
            img_size (int): image resizing size (ViT expects 224 usually)
        """
        self.img_size = img_size
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.samples = []  # list of tuples (image_path, label_str)
        self.class_to_idx = {}
        self.idx_to_class = {}

        if annotations_file:
            df = pd.read_csv(annotations_file)
            if "image_path" not in df.columns or "label" not in df.columns:
                raise ValueError("CSV must contain 'image_path' and 'label' columns.")
            for _, row in df.iterrows():
                imgp = str(row["image_path"])
                lbl = str(row["label"])
                self.samples.append((imgp, lbl))
        elif root_dir:
            # Walk folder structure
            classes = sorted(
                [
                    d
                    for d in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, d))
                ]
            )
            for c in classes:
                cpath = os.path.join(root_dir, c)
                for fname in os.listdir(cpath):
                    if fname.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".tiff", ".bmp")
                    ):
                        self.samples.append((os.path.join(cpath, fname), c))
        else:
            raise ValueError("Either annotations_file or root_dir must be provided.")

        # Build class index
        classes = sorted(list({lbl for _, lbl in self.samples}))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        if ensure_exists:
            missing = [p for p, _ in self.samples if not os.path.exists(p)]
            if missing:
                raise FileNotFoundError(
                    f"Found {len(missing)} missing images. Example: {missing[:3]}"
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_str = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        label_idx = self.class_to_idx[label_str]
        return img_tensor, label_idx, img_path

    def num_classes(self):
        return len(self.class_to_idx)

    def class_names(self):
        return [self.idx_to_class[i] for i in range(self.num_classes())]


if __name__ == "__main__":
    # quick sanity check usage
    ds = RajDataset(root_dir="../data/small_example", img_size=224, ensure_exists=False)
    print("Found classes:", ds.class_names())
    print("Length:", len(ds))
    x, y, p = ds[0]
    print("Sample shape:", x.shape, "label:", y, "path:", p)
