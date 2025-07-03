"""
eval_face_test.py

Evaluates a face verification model on the Task B TEST dataset using cosine similarity with FAISS.
Loads a trained ResNet50-based embedding model, builds embeddings for original and distorted images,
computes metrics (accuracy, precision, recall, macro-F1) at a fixed threshold, and logs the results.

Usage Example:
    python eval_face_test.py --data_path ../Comys_Hackathon5/Task_B/test
"""


import warnings
warnings.filterwarnings("ignore")  # suppress unnecessary warnings

import os
import argparse
import numpy as np
from PIL import Image
import faiss
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime
from tqdm import tqdm


# Configuration
# --------------
EMBEDDING_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCRIPT_NAME = "eval_face_test.py"
FIXED_THRESHOLD = 0.80  # similarity threshold


# Image transforms
# --------------
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize([0.5]*3, [0.5]*3),
    ToTensorV2()
])


# Model
# --------------
class StrongEmbeddingNet(nn.Module):
    """
    A CNN embedding model using a ResNet50 backbone for face matching.

    The model extracts features from an input image, projects them
    into a 512-dimensional embedding space, and normalizes the output
    to be used for cosine similarity computations.
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=False, num_classes=0)
        self.fc = nn.Linear(self.backbone.num_features, EMBEDDING_SIZE)

    def forward(self, x):
        x = self.backbone(x)
        return F.normalize(self.fc(x), p=2, dim=1)  # L2 normalize


# Build Task B data structure
# --------------
def build_taskB_data_structure(test_dir):
    """
    Builds a dictionary mapping each person to their original
    and valid distorted image paths.

    Args:
        test_dir (str): Path to the test dataset.

    Returns:
        dict: {identity: [list of image paths]}
    """
    data_structure = {'Task_B': {'test': {}}}
    person_folders = [p for p in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, p))]
    for person in person_folders:
        person_path = os.path.join(test_dir, person)
        data_structure['Task_B']['test'][person] = []
        original_images = [f for f in os.listdir(person_path) if f.endswith('.jpg') and f != 'distortion']
        for img_file in original_images:
            data_structure['Task_B']['test'][person].append(os.path.join(person_path, img_file))
            base_name = os.path.splitext(img_file)[0]
            distortion_path = os.path.join(person_path, 'distortion')
            for distortion_type in ['blurred', 'foggy', 'lowlight', 'noisy', 'rainy', 'resized', 'sunny']:
                distorted_filename = f"{base_name}_{distortion_type}.jpg"
                distorted_file_path = os.path.join(distortion_path, distorted_filename)
                if os.path.exists(distorted_file_path):
                    data_structure['Task_B']['test'][person].append(distorted_file_path)
    return data_structure


# Embedding generation
# --------------
def get_embedding(model, img_path):
    """
    Loads an image, applies preprocessing, runs it through the model,
    and returns a normalized embedding.

    Args:
        model (nn.Module): Trained embedding model.
        img_path (str): Image file path.

    Returns:
        np.ndarray: Normalized embedding vector.
    """
    image = np.array(Image.open(img_path).convert("RGB"))
    image = transform(image=image)['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model(image).cpu().numpy().squeeze()
    return (embedding / np.linalg.norm(embedding)).astype('float32')


# Evaluation function
# --------------
def evaluate_test_set(model, data_structure, threshold):
    """
    Evaluates model on the test dataset. Builds a gallery from original images,
    queries with distorted ones, computes metrics, prints, and logs them.

    Args:
        model (nn.Module): Loaded embedding model.
        data_structure (dict): Dictionary of image paths.
        threshold (float): Cosine similarity threshold.
    """
    model.eval()
    gallery_embeddings, gallery_labels = [], []

    # build gallery from original images only
    for person, images in data_structure['Task_B']['test'].items():
        for img_path in images:
            if "distortion" not in img_path:  # use only clean images
                emb = get_embedding(model, img_path)
                gallery_embeddings.append(emb)
                gallery_labels.append(person)

    gallery_embeddings = np.stack(gallery_embeddings).astype('float32')
    index = faiss.IndexFlatIP(gallery_embeddings.shape[1])
    index.add(gallery_embeddings)

    # verification on distorted images
    y_true, y_pred = [], []
    for person, images in tqdm(data_structure['Task_B']['test'].items(), desc="Evaluating TEST set"):
        distorted_images = [img for img in images if "distortion" in img]
        file_bar = tqdm(distorted_images, desc=f"  {person}", leave=False)
        for img_path in file_bar:
            file_bar.set_description(f"  {person} | {os.path.basename(img_path)}")
            query_emb = get_embedding(model, img_path).reshape(1, -1).astype('float32')
            sim, idx = index.search(query_emb, k=1)  # cosine similarity
            matched_label, score = gallery_labels[idx[0][0]], sim[0][0]
            y_true.append(1 if matched_label == person else 0)
            y_pred.append(1 if score > threshold else 0)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    timestamp = datetime.now()

    # print
    print(f"\n# [ {SCRIPT_NAME} | TEST SET | {timestamp} ]")
    print(f"Top-1 Accuracy      : {acc:.4f}")
    print(f"Macro-averaged F1   : {f1:.4f}")
    print(f"Precision           : {precision:.4f}")
    print(f"Recall              : {recall:.4f}")
    print(f"Threshold Used      : {threshold:.2f}")

    # write to file
    with open("metrics_output.txt", "a") as f:
        f.write(f"# [ {SCRIPT_NAME} | TEST SET | {timestamp} ]\n")
        f.write(f"Top-1 Accuracy      : {acc:.4f}\n")
        f.write(f"Macro-averaged F1   : {f1:.4f}\n")
        f.write(f"Precision           : {precision:.4f}\n")
        f.write(f"Recall              : {recall:.4f}\n")
        f.write(f"Threshold Used      : {threshold:.2f}\n\n")


# Main execution
# --------------
if __name__ == "__main__":
    """
    Loads model and test data, then runs evaluation to print
    and log verification metrics on the TEST set.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to test dataset (like train & val folders)")
    parser.add_argument('--model_path', type=str, default="FMM_ep3_f1_0.8832.pth", help="Path to trained model weights")
    args = parser.parse_args()

    data_structure_B = build_taskB_data_structure(args.data_path)
    model = StrongEmbeddingNet().to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    
    print(f"\nModel loaded from {args.model_path}")
    print(f"Using TEST DIR : {args.data_path}")
    print(f"Using Fixed Threshold : {FIXED_THRESHOLD:.2f}")

    print("\nStarting evaluation on TEST set...")
    evaluate_test_set(model, data_structure_B, FIXED_THRESHOLD)
