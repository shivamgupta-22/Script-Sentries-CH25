"""
eval_face_t_v.py

This script builds the structured data dictionary for Task B (face matching),
loads the trained embedding model, and evaluates it on TRAIN and VALIDATION datasets.

It prints and logs metrics (Accuracy, Precision, Recall, F1, Threshold)
to 'metrics_output.txt'.

Usage:
    python eval_face_t_v.py
"""

import warnings
warnings.filterwarnings("ignore")  # suppress warnings for cleaner output

import os, numpy as np
from PIL import Image
import faiss
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from datetime import datetime

# Configuration
# ------------------------
TRAIN_DIR = "../Comys_Hackathon5/Task_B/train"
VAL_DIR   = "../Comys_Hackathon5/Task_B/val"
MODEL_PATH = "FMM_ep3_f1_0.8832.pth"  # trained model checkpoint
EMBEDDING_SIZE = 512                  # output feature size
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCRIPT_NAME = "eval_face_t_v.py"
USER_THRESHOLD = 0.80  #similarity threshold

# Image transforms
# ----------------
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize([0.5]*3, [0.5]*3),
    ToTensorV2()
])

# Model definition
# ----------------
class StrongEmbeddingNet(nn.Module):
    """
    A CNN embedding model for face matching using ResNet50.

    The network extracts features from images with ResNet50,
    projects them to a smaller embedding size using a linear layer,
    and normalizes the output for cosine / inner product similarity.

    Attributes:
        backbone: The ResNet50 feature extractor.
        fc: Linear layer that reduces to EMBEDDING_SIZE.

    Methods:
        forward(x):
            Computes normalized embeddings from input images.
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=False, num_classes=0)  # remove classifier
        self.fc = nn.Linear(self.backbone.num_features, EMBEDDING_SIZE)  # project to 512

    def forward(self, x):
        x = self.backbone(x)
        return F.normalize(self.fc(x), p=2, dim=1)  # L2 normalize for cosine similarity

# Build Task B data structure
# ---------------------------
def build_taskB_data_structure(train_dir, val_dir):
    """
    Builds a structured dictionary for Task B face verification dataset.

    Organizes image paths under 'train' and 'val' splits, mapping each person
    to a list of their image file paths, including original and valid distortions.

    Args:
        train_dir (str): Path to the training dataset directory.
        val_dir (str): Path to the validation dataset directory.

    Returns:
        dict: Nested dictionary with structure:
              {'Task_B': {'train': {person: [image_paths]}, 'val': {...}}}
    """
    data_structure_B = {'Task_B': {'train': {}, 'val': {}}}
    for split, split_path in zip(['train', 'val'], [train_dir, val_dir]):
        person_folders = [p for p in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, p))]
        for person in person_folders:
            person_path = os.path.join(split_path, person)
            data_structure_B['Task_B'][split][person] = []
            original_images = [f for f in os.listdir(person_path) if f.endswith('.jpg') and f != 'distortion']
            for img_file in original_images:
                data_structure_B['Task_B'][split][person].append(os.path.join(person_path, img_file))
                base_name = os.path.splitext(img_file)[0]
                distortion_path = os.path.join(person_path, 'distortion')
                # include all distortion variants if they exist
                for distortion_type in ['blurred', 'foggy', 'lowlight', 'noisy', 'rainy', 'resized', 'sunny']:
                    distorted_filename = f"{base_name}_{distortion_type}.jpg"
                    distorted_file_path = os.path.join(distortion_path, distorted_filename)
                    if os.path.exists(distorted_file_path):
                        data_structure_B['Task_B'][split][person].append(distorted_file_path)
    return data_structure_B

# Embedding extraction
# --------------------
def get_embedding(model, img_path):
    """
    Generates a normalized embedding vector for a given image.

    Loads the image, applies preprocessing transforms, and passes it
    through the embedding model to obtain its feature representation.

    Args:
        model (torch.nn.Module): Trained embedding model.
        img_path (str): Path to the image file.

    Returns:
        np.ndarray: Normalized embedding vector (float32) of shape (embedding_dim,).
    """
    image = np.array(Image.open(img_path).convert("RGB"))
    image = transform(image=image)['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model(image).cpu().numpy().squeeze()
    return (embedding / np.linalg.norm(embedding)).astype('float32')

# Evaluation routine
# ------------------
def evaluate_with_threshold(model, data_structure, split, threshold):
    """
    Evaluates face verification accuracy using a fixed similarity threshold.

    Builds a FAISS index from original images (gallery), then queries it with
    distorted images to compute predictions. Calculates metrics including
    accuracy, precision, recall, and F1, and logs them to a text file.

    Args:
        model (torch.nn.Module): Trained embedding model.
        data_structure (dict): Task B data dictionary mapping persons to image paths.
        split (str): Either 'train' or 'val' to indicate which data split to use.
        threshold (float): Similarity score threshold for positive verification.

    Returns:
        None: Prints metrics to console and appends them to 'metrics_output.txt'.
    """
    model.eval()
    gallery_embeddings, gallery_labels = [], []

    # build gallery from original images only
    for person, images in data_structure['Task_B'][split].items():
        for img_path in images:
            if "distortion" not in img_path:
                emb = get_embedding(model, img_path)
                gallery_embeddings.append(emb)
                gallery_labels.append(person)

    gallery_embeddings = np.stack(gallery_embeddings).astype('float32')
    index = faiss.IndexFlatIP(gallery_embeddings.shape[1])  # cosine similarity index
    index.add(gallery_embeddings)

    # evaluate on distorted images
    y_true, y_pred = [], []
    for person, images in tqdm(data_structure['Task_B'][split].items(), desc=f"Evaluating {split}"):
        distorted_images = [img for img in images if "distortion" in img]
        file_bar = tqdm(distorted_images, desc=f"  {person}", leave=False)
        for img_path in file_bar:
            file_bar.set_description(f"  {person} | {os.path.basename(img_path)}")
            query_emb = get_embedding(model, img_path).reshape(1, -1).astype('float32')
            sim, idx = index.search(query_emb, k=1)
            matched_label, score = gallery_labels[idx[0][0]], sim[0][0]
            y_true.append(1 if matched_label == person else 0)
            y_pred.append(1 if score > threshold else 0)

    # compute metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    timestamp = datetime.now()

    # print
    print(f"\n# [ {SCRIPT_NAME} | {split.upper()} SET | {timestamp} ]")
    print(f"Top-1 Accuracy      : {acc:.4f}")
    print(f"Macro-averaged F1   : {f1:.4f}")
    print(f"Precision           : {precision:.4f}")
    print(f"Recall              : {recall:.4f}")
    print(f"Threshold Used      : {threshold:.2f}")

    # write
    with open("metrics_output.txt", "a") as f:
        f.write(f"# [ {SCRIPT_NAME} | {split.upper()} SET | {timestamp} ]\n")
        f.write(f"Top-1 Accuracy      : {acc:.4f}\n")
        f.write(f"Macro-averaged F1   : {f1:.4f}\n")
        f.write(f"Precision           : {precision:.4f}\n")
        f.write(f"Recall              : {recall:.4f}\n")
        f.write(f"Threshold Used      : {threshold:.2f}\n\n")

# Main execution
# --------------
if __name__ == "__main__":
    """
    Loads data, model, and evaluates on train & val sets.
    """
    data_structure_B = build_taskB_data_structure(TRAIN_DIR, VAL_DIR)  # load dataset structure
    model = StrongEmbeddingNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  # load model weights
    print(f"\nModel loaded from {MODEL_PATH}")
    print(f"Using TRAIN DIR : {TRAIN_DIR}")
    print(f"Using VAL   DIR : {VAL_DIR}")

    print("\nStarting evaluation on TRAIN set...")
    evaluate_with_threshold(model, data_structure_B, 'train', USER_THRESHOLD)

    print("\nStarting evaluation on VAL set...")
    evaluate_with_threshold(model, data_structure_B, 'val', USER_THRESHOLD)
