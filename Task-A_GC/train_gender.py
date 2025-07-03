"""
train_gender.py

This script trains a gender classifier on the provided dataset. 
It performs the following:

1. Prints dataset statistics such as number of images in each split and class.
2. Trains an EfficientNetV2-based gender classifier.
3. Logs training and validation metrics (Loss, Accuracy, F1) for each epoch 
   to the console and to a file named 'training_stats.txt'.
4. Also logs metrics to TensorBoard for easy visualization.

Usage:
    python train_gender.py
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

# Paths and logging
# -----------------
train_dir = "../Comys_Hackathon5/Task_A/train/"
val_dir = "../Comys_Hackathon5/Task_A/val/"
log_file = open("training_stats.txt", "w", encoding="utf-8")

def log_print(msg):
    """
    Prints a message to the console and also writes it to the 
    training_stats.txt log file.
    """
    print(msg)
    log_file.write(msg + "\n")

# Print dataset stats
# -------------------
def print_dataset_stats():
    """
    Prints the dataset statistics for both the training and validation splits. 
    Shows the number of images in each class (male, female) and total count.
    Useful to verify class distribution and detect imbalance.
    """
    log_print("Dataset Statistics:")
    for split, path in [("Train", train_dir), ("Validation", val_dir)]:
        male_path = os.path.join(path, "male")
        female_path = os.path.join(path, "female")
        # count images in each class
        male_count = len([f for f in os.listdir(male_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        female_count = len([f for f in os.listdir(female_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total = male_count + female_count
        log_print(f"\n{split}:")
        log_print(f"    Male images   : {male_count}")
        log_print(f"    Female images : {female_count}")
        log_print(f"    Total images  : {total}")

# Build data structure with file paths
# ------------------------------------
data_structure = {
    'Task_A': {
        'train': {
            'male': [os.path.join(train_dir, 'male', f)
                     for f in os.listdir(os.path.join(train_dir, 'male'))
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
            'female': [os.path.join(train_dir, 'female', f)
                       for f in os.listdir(os.path.join(train_dir, 'female'))
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        },
        'val': {
            'male': [os.path.join(val_dir, 'male', f)
                     for f in os.listdir(os.path.join(val_dir, 'male'))
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
            'female': [os.path.join(val_dir, 'female', f)
                       for f in os.listdir(os.path.join(val_dir, 'female'))
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        }
    }
}

# Dataset class
# -------------
class GenderDataset(Dataset):
    """
    Custom PyTorch Dataset for gender classification. It loads 
    image file paths and labels (0 for male, 1 for female).

    Args:
        data_dict (dict): Dictionary with 'male' and 'female' keys, 
                          each containing list of image file paths.
        transform (callable, optional): Albumentations transformations 
                                        applied to each image.

    Returns:
        tuple: Transformed image tensor and the corresponding label.
    """
    def __init__(self, data_dict, transform=None):
        self.transform = transform
        self.samples = []
        # build list of (path, label) pairs
        for img_path in data_dict['male']:
            self.samples.append((img_path, 0))
        for img_path in data_dict['female']:
            self.samples.append((img_path, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # load image and convert to numpy array (H, W, C)
        img = np.array(Image.open(img_path).convert('RGB'))
        if self.transform:
            # apply augmentations and normalization
            img = self.transform(image=img)['image']
        return img, torch.tensor(label, dtype=torch.float32)

# Image augmentations
# -------------------
train_transform = A.Compose([
    A.Resize(256, 256), 
    A.RandomCrop(224, 224), 
    A.HorizontalFlip(p=0.5), 
    A.RandomBrightnessContrast(p=0.2), 
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5), # slight affine
    A.CoarseDropout(max_holes=1, max_height=32, max_width=32,
                    min_holes=1, min_height=32, min_width=32, p=0.5), # random erasing
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ToTensorV2() 
])

val_transform = A.Compose([
    A.Resize(256, 256), 
    A.CenterCrop(224, 224), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Gender classification model
# ---------------------------
class GenderClassifier(nn.Module):
    """
    Gender classification model using an EfficientNetV2-S backbone.
    The final classifier head outputs a single logit indicating 
    the probability of being female.

    Male is represented as class 0, female as class 1.
    """
    def __init__(self):
        super().__init__()
        # load EfficientNetV2-S pretrained
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=True)
        in_features = self.backbone.classifier.in_features
        # replace final classifier to output a single value
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.backbone(x)

# Get data loaders
# ----------------
def get_loaders(batch_size=32, num_workers=2):
    """
    Creates PyTorch DataLoader instances for the training and validation datasets.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker processes for data loading.

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_set = GenderDataset(data_structure['Task_A']['train'], transform=train_transform)
    val_set = GenderDataset(data_structure['Task_A']['val'], transform=val_transform)
    # shuffle only for train
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

# Training function
# -----------------
def train_model(model, train_loader, val_loader, device, num_epochs=20):
    """
    Trains the model on the training data and evaluates it on the validation data.
    Logs Loss, Accuracy, and F1-Score for each epoch to both console and file.

    Also writes metrics to TensorBoard for visualization.
    """
    # use pos_weight to adjust for imbalance
    pos_weight = torch.tensor([1532 / 394], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler = torch.amp.GradScaler("cuda")
    writer = SummaryWriter()  # for TensorBoard logging

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0
        all_preds, all_labels = [], []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model(images).squeeze()
                # compute BCE loss
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # convert logits to 0/1 predictions using sigmoid
            preds = (torch.sigmoid(outputs) > 0.5).float()
            running_loss += loss.item() * images.size(0)
            running_corrects += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects / len(train_loader.dataset)
        train_f1 = f1_score(all_preds, all_labels)

        # run evaluation on validation set
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, device, criterion)

        # print and log metrics
        msg = (f"\nEpoch {epoch+1}:\n"
               f"Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}\n"
               f"Val   -> Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        log_print(msg)

        # prompt to optionally save the model
        save_prompt = f"Save model as ep{epoch+1}_f1_{val_f1:.4f}.pth? (y/n): "
        save_choice = input(save_prompt).strip().lower()
        log_file.write(f"{save_prompt}{save_choice}\n")
        if save_choice == 'y':
            filename = f'GCM_ep{epoch+1}_f1_{val_f1:.4f}.pth'
            torch.save(model.state_dict(), filename)
            log_print(f"Model saved as {filename}")
        else:
            log_print("Model not saved for this epoch.")    

        # tensorBoard logs
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("F1/train", train_f1, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)

        # step LR scheduler
        scheduler.step(epoch + len(train_loader)/len(train_loader))

    writer.close()

# Evaluate on validation
# ----------------------
def evaluate_model(model, loader, device, criterion):
    """
    Evaluates the model on a given dataset loader. Computes 
    average Loss, Accuracy, and F1-Score.
    """
    model.eval()
    running_loss, running_corrects = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            # update stats
            running_loss += loss.item() * images.size(0)
            running_corrects += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss = running_loss / len(loader.dataset)
    acc = running_corrects / len(loader.dataset)
    f1 = f1_score(all_preds, all_labels)
    return loss, acc, f1

# Main
# ----
if __name__ == "__main__":
    print_dataset_stats()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GenderClassifier().to(device)
    train_loader, val_loader = get_loaders()
    log_print("\nStarting training...")
    train_model(model, train_loader, val_loader, device)
    log_file.close()