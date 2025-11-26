# eval_raj_vit.py
import argparse
import csv

import torch
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from raj_dataset import RajDataset

try:
    from sklearn.metrics import classification_report, confusion_matrix

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    print(
        "sklearn not available — will compute basic accuracy only. (pip install scikit-learn for full report)"
    )


def load_vit_model(num_classes, device, checkpoint_path):
    try:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        in_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(in_features, num_classes)
    except Exception:
        model = models.vit_b_16(pretrained=True)
        in_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(in_features, num_classes)

    model = model.to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations", type=str, default=None, help="CSV with image_path,label"
    )
    parser.add_argument(
        "--root_dir", type=str, default=None, help="root folder with class subfolders"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="path to model state (vit_best.pth or full ckpt)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--output_csv", type=str, default="preds.csv")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Using device:", device)

    ds = RajDataset(
        root_dir=args.root_dir,
        annotations_file=args.annotations,
        img_size=args.img_size,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = load_vit_model(ds.num_classes(), device, args.checkpoint)
    print("Loaded model and dataset. Running inference...")

    all_preds = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, ncols=120):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_paths.extend(paths)

    # save CSV
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_path",
                "true_label",
                "pred_label",
                "true_label_str",
                "pred_label_str",
            ]
        )
        for p, t, pr in zip(all_paths, all_labels, all_preds):
            writer.writerow(
                [p, int(t), int(pr), ds.idx_to_class[int(t)], ds.idx_to_class[int(pr)]]
            )

    print(f"Saved predictions to {args.output_csv}")

    if SKLEARN_AVAILABLE:
        print("Classification report:")
        print(
            classification_report(
                all_labels, all_preds, target_names=ds.class_names(), digits=4
            )
        )
        print("Confusion matrix:")
        print(confusion_matrix(all_labels, all_preds))
    else:
        # fallback basic accuracy
        correct = sum([1 if a == b else 0 for a, b in zip(all_preds, all_labels)])
        total = len(all_labels)
        print(f"Accuracy: {correct}/{total} = {correct/total:.4f}")
        print("Install scikit-learn for nicer reports: pip install scikit-learn")


if __name__ == "__main__":
    main()
