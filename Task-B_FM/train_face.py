"""
train_face.py

This script trains a face matching model on Task B datasets using triplet loss. 
It performs the following:

1. Prints dataset statistics (number of identities, original & distorted images).
2. Builds a structured data dictionary for Task B.
3. Trains a ResNet50-based embedding model using triplet margin loss and extensive data augmentation.
4. Periodically evaluates with FAISS + cosine similarity on the validation set, tuning thresholds for best macro-F1.
5. Logs training & validation metrics and saves model checkpoints.

Usage:
    python train_face.py
"""


import os, random, numpy as np
from PIL import Image
from tqdm import tqdm
import faiss
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings("ignore")


# Configuration
# -------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DATA_DIR = "../Comys_Hackathon5/Task_B"
TRAIN_DIR = "../Comys_Hackathon5/Task_B/train"
VAL_DIR   = "../Comys_Hackathon5/Task_B/val"

BATCH_SIZE = 64
EPOCHS = 15
EMBEDDING_SIZE = 512
THRESHOLDS = np.linspace(0.5, 0.9, 9)  # to find best threshold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# simple logger that also writes to file
def log_and_print(msg, filename="training_stats.txt"):
    print(msg)
    with open(filename, "a") as f:
        f.write(msg + "\n")


# Transforms
# -------------------
# for training (heavy augmentation)
transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.MotionBlur(p=0.2),
    A.GaussianBlur(p=0.1),
    A.CoarseDropout(p=0.2),
    A.Normalize([0.5]*3, [0.5]*3),
    ToTensorV2(),
])

# for evaluation embeddings (only resize & normalize)
simple_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize([0.5]*3, [0.5]*3),
    ToTensorV2(),
])


# Dataset statistics & structure
# -------------------
def print_taskB_dataset_stats(base_dir):
    """
    Prints summary statistics of the Task B dataset.

    Shows the number of identities, original images,
    distorted images, and total images for both
    train and validation splits.
    """
    log_and_print("Task B - Face Recognition Dataset Statistics:")
    for split in ['train', 'val']:
        split_path = os.path.join(base_dir, split)
        person_folders = [p for p in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, p))]
        total_original = 0
        total_distorted = 0

        for person in person_folders:
            person_path = os.path.join(split_path, person)
            distortion_path = os.path.join(person_path, 'distortion')

            original_imgs = [f for f in os.listdir(person_path)
                             if f.endswith('.jpg') and not f.startswith('.') and f != 'distortion']
            total_original += len(original_imgs)

            if os.path.exists(distortion_path):
                distorted_imgs = [f for f in os.listdir(distortion_path)
                                  if f.endswith('.jpg') and not f.startswith('.')]
                total_distorted += len(distorted_imgs)

        log_and_print(f"\n{split.capitalize()} Set:")
        log_and_print(f"    Total identities       : {len(person_folders)}")
        log_and_print(f"    Total original images  : {total_original}")
        log_and_print(f"    Total distorted images : {total_distorted}")
        log_and_print(f"    Total all images       : {total_original + total_distorted}")


def build_taskB_data_structure(base_dir):
    """
    Builds a nested dictionary mapping each identity to their image paths.

    Organizes the train and validation sets, associating
    each person with their original and valid distorted images.

    Returns:
        dict: Structured data dictionary for Task B.
    """
    log_and_print("\nBuilding structured data dictionary for Task B...")
    data_structure_B = {'Task_B': {'train': {}, 'val': {}}}
    for split in ['train', 'val']:
        split_path = os.path.join(base_dir, split)
        person_folders = [p for p in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, p))]
        for person in person_folders:
            person_path = os.path.join(split_path, person)
            data_structure_B['Task_B'][split][person] = []
            original_images = [f for f in os.listdir(person_path) if f.endswith('.jpg') and f != 'distortion']
            for img_file in original_images:
                data_structure_B['Task_B'][split][person].append(os.path.join(person_path, img_file))
                base_name = os.path.splitext(img_file)[0]
                distortion_path = os.path.join(person_path, 'distortion')
                for distortion_type in ['blurred', 'foggy', 'lowlight', 'noisy', 'rainy', 'resized', 'sunny']:
                    distorted_filename = f"{base_name}_{distortion_type}.jpg"
                    distorted_file_path = os.path.join(distortion_path, distorted_filename)
                    if os.path.exists(distorted_file_path):
                        data_structure_B['Task_B'][split][person].append(distorted_file_path)
    log_and_print("\nData structure build complete!")
    log_and_print(f"Total Task B training identities : {len(data_structure_B['Task_B']['train'])}")
    log_and_print(f"Total Task B validation identities : {len(data_structure_B['Task_B']['val'])}")
    return data_structure_B


# Triplet dataset class
# -------------------
class TripletFaceDataset(Dataset):
    """
    Custom dataset for generating triplets (anchor, positive, negative).

    Samples triplets on-the-fly for triplet loss training,
    applying the specified transformations to each image.
    """
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
         # build mapping: identity -> all images (original + distortions)
        for identity in sorted(os.listdir(root_dir)):
            id_path = os.path.join(root_dir, identity)
            originals = [f for f in os.listdir(id_path) if f.endswith('.jpg')]
            distorted = []
            for f in originals:
                for i in range(1, 8):
                    distorted_path = f"distortion/{f[:-4]}_{i}.jpg"
                    full_path = os.path.join(id_path, distorted_path)
                    if os.path.exists(full_path):
                        distorted.append(distorted_path)
            all_imgs = originals + distorted
            if len(all_imgs) >= 2:  # ensure at least anchor+positive
                self.samples.append((identity, all_imgs))

    def __len__(self):
        return 64000  # fixed large number to allow many random samples

    def __getitem__(self, idx):
        anchor_person, anchor_imgs = random.choice(self.samples)
        anchor_img = random.choice(anchor_imgs)
        positive_candidates = [img for img in anchor_imgs if img != anchor_img]
        if not positive_candidates:
            return self.__getitem__(idx)  # try again
        positive_img = random.choice(positive_candidates)
        # sample negative from different person
        negative_person, negative_imgs = random.choice([s for s in self.samples if s[0] != anchor_person])
        negative_img = random.choice(negative_imgs)
        return (
            self.load_image(anchor_person, anchor_img),
            self.load_image(anchor_person, positive_img),
            self.load_image(negative_person, negative_img),
        )

    def load_image(self, identity, filename):
        img_path = os.path.join(self.root_dir, identity, filename)
        img = Image.open(img_path).convert("RGB")
        return self.transform(image=np.array(img))['image']


# Model definition
# -------------------
class StrongEmbeddingNet(nn.Module):
    """
    A CNN embedding model for face matching using ResNet50.

    Extracts features using a pre-trained ResNet50, projects
    to EMBEDDING_SIZE, and normalizes the output vector.
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=True, num_classes=0)
        self.fc = nn.Linear(self.backbone.num_features, EMBEDDING_SIZE)

    def forward(self, x):
        x = self.backbone(x)
        return F.normalize(self.fc(x), p=2, dim=1)


# Embedding function
# -------------------
def get_embedding(model, img_path):
    """
    Generates a normalized embedding for a given image.

    Loads the image, applies simple preprocessing, and
    computes its embedding using the model.
    """
    img = Image.open(img_path).convert("RGB")
    img = simple_transform(image=np.array(img))['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        return model(img).cpu().numpy().squeeze()


# Training function
# -------------------
def train_epoch(model, loader, optimizer, criterion, scaler):
    """
    Trains the embedding model for one epoch using triplet loss.

    Computes anchor-positive and anchor-negative distances,
    scales gradients for mixed precision, and updates weights.

    Returns average loss, average positive distance, and
    average negative distance.
    """
    model.train()
    total_loss, pos_dists, neg_dists = 0, [], []
    for a, p, n in tqdm(loader, desc="Training"):
        a, p, n = a.to(device), p.to(device), n.to(device)
        optimizer.zero_grad()
        with autocast():  # automatic mixed precision
            ea, ep, en = model(a), model(p), model(n)
            pos_dist = F.pairwise_distance(ea, ep)
            neg_dist = F.pairwise_distance(ea, en)
            loss = criterion(ea, ep, en)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        pos_dists.append(pos_dist.mean().item())
        neg_dists.append(neg_dist.mean().item())
    return total_loss / len(loader), np.mean(pos_dists), np.mean(neg_dists)


# Evaluation function
# -------------------
def tune_and_print_metrics(model, directory, thresholds=THRESHOLDS):
    """
    Evaluates model using cosine similarity with FAISS,
    tuning thresholds to maximize macro-F1 on validation set.

    Prints metrics and returns the best threshold and F1.
    """
    model.eval()  # switch model to evaluation mode
    gallery_embeddings, gallery_labels = [], []
    for identity in os.listdir(directory):
        id_path = os.path.join(directory, identity)
        for f in os.listdir(id_path):
            if f.endswith(".jpg"):  # only gallery (original) images
                emb = get_embedding(model, os.path.join(id_path, f))
                gallery_embeddings.append(emb / np.linalg.norm(emb))  # L2 normalize
                gallery_labels.append(identity)
    gallery_embeddings = np.stack(gallery_embeddings).astype('float32')
    index = faiss.IndexFlatIP(gallery_embeddings.shape[1])
    index.add(gallery_embeddings)

    best_f1, best_thresh, best_stats = 0, 0, ()
    for threshold in thresholds:
        y_true, y_pred = [], []
        # for each distorted image, find top-1 match in gallery
        for identity in os.listdir(directory):
            distort_path = os.path.join(directory, identity, "distortion")
            if not os.path.exists(distort_path): continue
            for f in os.listdir(distort_path):
                query_emb = get_embedding(model, os.path.join(distort_path, f))
                query_emb = (query_emb / np.linalg.norm(query_emb)).reshape(1, -1).astype('float32')
                sim, idx = index.search(query_emb, k=1)  # top-1 search
                matched_label, score = gallery_labels[idx[0][0]], sim[0][0]
                y_true.append(1 if matched_label == identity else 0)
                y_pred.append(1 if score > threshold else 0)
        # compute metrics
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, threshold
            best_stats = (acc, precision, recall, f1)
    acc, precision, recall, f1 = best_stats
    msg = f"Validation Metrics: Acc={acc:.4f} | Prec={precision:.4f} | Recall={recall:.4f} | Macro-F1={f1:.4f} @ Thresh={best_thresh:.2f}"
    log_and_print(msg)
    return best_thresh, best_f1


# Main execution
# -------------------
if __name__ == "__main__":
    """
    Loads data, initializes model, trains with triplet loss,
    and periodically evaluates on the validation set.
    """
    print_taskB_dataset_stats(DATA_DIR)
    data_structure_B = build_taskB_data_structure(DATA_DIR)

    model = StrongEmbeddingNet().to(device, memory_format=torch.channels_last)
    train_loader = DataLoader(
        TripletFaceDataset(TRAIN_DIR, transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.TripletMarginLoss(margin=0.3)
    scaler = GradScaler()

    log_and_print("\nStarting training...")
    for epoch in range(EPOCHS):
        log_and_print(f"\nEpoch {epoch+1}/{EPOCHS}:")
        # train for one epoch: returns average triplet loss, avg positive & negative distances
        loss, avg_pos, avg_neg = train_epoch(model, train_loader, optimizer, criterion, scaler)
        log_and_print(f"Loss: {loss:.4f} | PosDist: {avg_pos:.4f} | NegDist: {avg_neg:.4f}")
        
        # every 3 epochs, validate using FAISS nearest neighbor search on distortions
        # finds best threshold for macro-F1, then saves checkpoint
        if (epoch+1) % 3 == 0:
            val_thresh, val_f1 = tune_and_print_metrics(model, VAL_DIR)
            torch.save(model.state_dict(), f"FMM_ep{epoch+1}_f1_{val_f1:.4f}.pth")
            log_and_print(f"Saved checkpoint: FMM_ep{epoch+1}_f1_{val_f1:.4f}.pth")
