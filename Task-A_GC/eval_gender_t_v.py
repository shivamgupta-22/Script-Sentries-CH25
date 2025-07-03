"""
eval_gender_t_v.py

This script loads the trained gender classifier model
and evaluates it on TRAIN and VALIDATION datasets (hardcoded paths).

It prints and appends metrics (Accuracy, Precision, Recall, F1)
to a file named 'metrics_output.txt' in a clean format like:
[ eval_gender_t_v.py | TRAIN SET | timestamp ]
[ eval_gender_t_v.py | VALIDATION SET | timestamp ]
"""

import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime

# Data loading class
# ------------------
class GenderDataset(Dataset):
    """
    Dataset for loading gender classification images.

    Args:
        data_paths (dict): Dictionary with 'male' and 'female' keys pointing to lists of image paths.
        transform (callable, optional): Albumentations transform to apply on images.

    Returns:
        tuple: (transformed image tensor, label)
    """
    def __init__(self, data_paths, transform=None):
        self.samples = []
        self.transform = transform
        for img_path in data_paths['male']:
            self.samples.append((img_path, 0))  # 0 -> male
        for img_path in data_paths['female']:
            self.samples.append((img_path, 1))  # 1 -> female

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label

# Image preprocessing
# -------------------
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
    Gender classification model using EfficientNetV2.

    Outputs a single logit for binary classification (male/female).
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=False)
        in_features = self.backbone.classifier.in_features
        # replace classifier head with custom layers
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)  # binary output
        )

    def forward(self, x):
        return self.backbone(x)

# Build dataset file paths
# ------------------------
def build_image_paths(root_dir):
    """
    Builds dictionary of image file paths for male and female.

    Args:
        root_dir (str): Path to dataset directory containing 'male' and 'female' folders.

    Returns:
        dict: {'male': list of male image paths, 'female': list of female image paths}
    """
    return {
        'male': [os.path.join(root_dir, 'male', f)
                 for f in os.listdir(os.path.join(root_dir, 'male'))
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        'female': [os.path.join(root_dir, 'female', f)
                   for f in os.listdir(os.path.join(root_dir, 'female'))
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    }

# Dataloader creation
# -------------------
def create_data_loader(data_paths, batch_size=32, num_workers=2):
    """
    Creates DataLoader for gender dataset.

    Args:
        data_paths (dict): Dictionary of image paths.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        DataLoader: PyTorch DataLoader for dataset.
    """
    dataset = GenderDataset(data_paths, transform=val_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Evaluation routine
# ------------------
def evaluate_model(model, loader, device, label, file, script_name):
    """
    Evaluates the model on a dataset, computes metrics (Accuracy, Precision,
    Recall, F1), prints them, and appends to a log file.

    Args:
        model (nn.Module): Trained model.
        loader (DataLoader): Data loader for dataset.
        device (torch.device): CPU or CUDA device.
        label (str): Dataset label for logs (e.g., 'TRAIN SET').
        file (file object): File handle to write metrics.
        script_name (str): Name of the script for log header.
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

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    correct = (np.array(all_preds) == np.array(all_labels)).sum()
    total = len(all_labels)
    wrong = total - correct

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
# -----------
if __name__ == "__main__":
    script_name = "eval_gender_t_v.py"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hardcoded paths
    train_data_path = "../Comys_Hackathon5/Task_A/train"
    val_data_path   = "../Comys_Hackathon5/Task_A/val"
    model_path      = "GCM_ep4_f1_0.9577.pth"

    # load trained weights
    model = GenderClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"\nModel loaded from {model_path}\n")

    # prepare datasets and loaders
    train_structure = build_image_paths(train_data_path)
    val_structure   = build_image_paths(val_data_path)

    train_loader = create_data_loader(train_structure)
    val_loader   = create_data_loader(val_structure)

    # evaluate and log metrics
    with open("metrics_output.txt", "a") as file:
        evaluate_model(model, train_loader, device, label="TRAIN SET", file=file, script_name=script_name)
        evaluate_model(model, val_loader, device, label="VALIDATION SET", file=file, script_name=script_name)
