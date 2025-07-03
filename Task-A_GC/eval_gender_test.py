"""
eval_gender_test.py

This script loads the trained gender classifier model
and evaluates it on a TEST dataset provided by the user.

Usage Example:
    python eval_gender_test.py --data_path ../Comys_Hackathon5/Task_A/test

It prints and appends metrics (Accuracy, Precision, Recall, F1)
to a file named 'metrics_output.txt' in a clean format like:
[ eval_gender_test.py | TEST SET | timestamp ]
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Data loading class
# ------------------
class GenderDataset(Dataset):
    """
    PyTorch Dataset for loading gender classification images.

    Args:
        data_dict (dict): Dictionary with 'male' and 'female' keys containing image paths.
        transform (callable, optional): Albumentations transform for preprocessing.

    Returns:
        tuple: (transformed image tensor, label)
    """
    def __init__(self, data_dict, transform=None):
        self.samples = []
        self.transform = transform
        # store (file_path, label) pairs
        for img_path in data_dict['male']:
            self.samples.append((img_path, 0))
        for img_path in data_dict['female']:
            self.samples.append((img_path, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = np.array(Image.open(img_path).convert('RGB'))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label

# Image transforms
# ----------------
# Resize, crop and normalize images for EfficientNet
val_transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Gender classification model
# ---------------------------
class GenderClassifier(nn.Module):
    """
    Gender classification model using EfficientNetV2 backbone.

    Outputs:
        Single sigmoid logit indicating male (0) or female (1).
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=False)
        # replace classifier layer
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.backbone(x)

# Build data structure
# --------------------
def build_data_structure(root):
    """
    Creates dictionary with lists of image file paths under 'male' and 'female' keys.

    Args:
        root (str): Path to dataset folder containing 'male' and 'female' subfolders.

    Returns:
        dict: {'male': [...], 'female': [...]}
    """
    return {
        'male': [os.path.join(root, 'male', f)
                 for f in os.listdir(os.path.join(root, 'male'))
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        'female': [os.path.join(root, 'female', f)
                   for f in os.listdir(os.path.join(root, 'female'))
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    }

# Get data loader
# ---------------
def get_loader(data_structure, batch_size=32, num_workers=2):
    """
    Creates DataLoader for gender dataset.

    Args:
        data_structure (dict): Dictionary of image paths.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for loading.

    Returns:
        DataLoader: PyTorch DataLoader for dataset.
    """
    dataset = GenderDataset(data_structure, transform=val_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Evaluation function
# -------------------
def evaluate_model(model, loader, device, label, file, script_name):
    """
    Evaluates the model on given data, prints and logs metrics.

    Args:
        model (torch.nn.Module): Trained gender classifier.
        loader (DataLoader): DataLoader with data to evaluate.
        device (torch.device): CUDA or CPU device.
        label (str): Name of dataset (for logs).
        file (file object): Open file handle to append metrics.
        script_name (str): Name of this script for header logs.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():  # no gradient needed for eval
        for images, labels in tqdm(loader, desc=f"Evaluating on {label}"):
            images = images.to(device)
            labels = labels.float().to(device)
            outputs = model(images).squeeze()
            preds = (torch.sigmoid(outputs) > 0.5).float()  # threshold at 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    correct = (np.array(all_preds) == np.array(all_labels)).sum()
    total = len(all_labels)
    wrong = total - correct

    # print and log results
    header = f"# [ {script_name} | {label} | {datetime.now()} ]\n"
    metrics = (
        f"Accuracy            : {acc:.4f}\n"
        f"Precision           : {prec:.4f}\n"
        f"Recall              : {rec:.4f}\n"
        f"F1-Score            : {f1:.4f}\n"
        f"Correct Predictions : {correct}/{total}\n"
        f"Wrong Predictions   : {wrong}/{total}\n"
    )
    print(header + metrics)
    file.write(header + metrics + "\n")

# Main
# ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to test data folder with 'male' and 'female' subfolders.")
    parser.add_argument("--model_path", type=str, default="GCM_ep4_f1_0.9577.pth",
                        help="Path to trained model weights file.")
    args = parser.parse_args()

    script_name = "eval_gender_test.py"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load trained weights
    model = GenderClassifier().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"\nModel loaded from {args.model_path}\n")

    # prepare data loader
    test_structure = build_data_structure(args.data_path)
    test_loader = get_loader(test_structure)

    # run evaluation and append metrics to file
    with open("metrics_output.txt", "a") as file:
        evaluate_model(model, test_loader, device, label="TEST SET", file=file, script_name=script_name)