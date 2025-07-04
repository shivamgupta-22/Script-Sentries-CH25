# COMSYS Hackathon-5, 2025

This repository provides a comprehensive solution to two computer vision challenges presented in **ComsysHackathon 25**, held under the **6th International Conference on Frontiers in Computing and Systems (COMSYS 2025)**.

---

## Team Details

**Team Name:** Script Sentries

**Team Members:** Shivam Gupta (*Team Leader*), Abhay Prasad Bari, Ranit Sarkhel

---

## Task A – Gender Classification (Binary)

### Objective

Design a deep learning-based classifier to predict the binary gender (Male/Female) from face images, ensuring high accuracy across diverse lighting, angle, and expression variations.

### Model Architecture

* **Backbone:** EfficientNetV2-S (from `timm` library, pretrained)
* **Classifier Head:**

  * `nn.Linear(in_features, 256)`
  * `nn.ReLU()`
  * `nn.Dropout(0.5)`
  * `nn.Linear(256, 1)`
  * Final activation: `Sigmoid`

### Dataset Layout

```
Task_A/
├── train/
│   ├── male/
│   └── female/
├── val/
└── test/
```

### Training Process

Run the following from inside `Task-A_GC/`:

```python
# train_gender.py
"""
Train a binary gender classification model using EfficientNetV2-S.
Saves model checkpoint and logs training stats.
"""
python train_gender.py
```

* Logs: `training_stats.txt`, TensorBoard logs (optional)
* Model saved as: `GCM_ep4_f1_0.9577.pth`

### Evaluation Instructions

#### Set Working Directory

Ensure that your terminal is inside the `Task-A_GC/` directory so relative paths resolve correctly:

```
root/
├── Task-A_GC/
│   ├── eval_gender_t_v.py
│   └── eval_gender_test.py
└── Comys_Hackathon5/
    └── Task_A/
        ├── train/
        ├── val/
        └── test/
```

```bash
cd Task-A_GC
```

#### Evaluate on TRAIN + VALIDATION

```bash
python eval_gender_t_v.py
```

This expects:

```
../Comys_Hackathon5/Task_A/train/
../Comys_Hackathon5/Task_A/val/
```

Each with:

```
male/
female/
```

#### Evaluate on TEST

```bash
python eval_gender_test.py --data_path ../Comys_Hackathon5/Task_A/test
```

#### Output Metrics

All results are printed and appended to `metrics_output.txt`. Example:

```
# [ evaluate_gender_test.py | EXAMPLE SET | 2025-06-30 20:56:03.002302 ]
Accuracy            : 0.9583
Precision           : 0.9472
Recall              : 0.9720
F1-Score            : 0.9594
Correct Predictions : 230/240
Wrong Predictions   : 10/240
```

---

### Notes

- Always run scripts inside the respective folders.
- Metrics result append to `metrics_output.txt`.
- `training_stats.txt` shows epoch-wise logs.
- `model_diagram.png` shows architecture flow.

---

## Task B – Face Recognition (Multi-class)

### Objective

Develop an embedding-based face recognition model robust to distortions such as blur, fog, low-light, and noise. The goal is identity verification using cosine similarity.

### Model Architecture

* **Backbone:** ResNet50 (classification head removed)
* **Feature Output:** 2048-D vector from final convolutional layer
* **Embedding Projection:**

  * `nn.Linear(2048, 512)` to reduce dimensionality
* **Activation:** None (to retain feature integrity)
* **Normalization:**

  * Apply L2 normalization to produce unit vectors
* **Output:**

  * 512-D normalized embeddings used for FAISS similarity search

### Dataset Layout

```
Task_B/
├── train/
│   ├── <person_id>/
│   │   └── distortion/
├── val/
└── test/
```

### Training Process

Execute the training inside `Task-B_FM/`:

```python
# train_face.py
"""
Train a ResNet50-based embedding model using triplet loss.
Triplets are dynamically sampled from original and distorted data.
"""
python train_face.py
```

* Logs stored in: `training_stats.txt`
* Best model saved as: `FMM_ep3_f1_0.8832.pth`

### Evaluation Instructions

#### Set Working Directory

Ensure your terminal is inside `Task-B_FM/` so paths resolve correctly:

```
root/
├── Task-B_FM/
│   ├── eval_face_t_v.py
│   └── eval_face_test.py
└── Comys_Hackathon5/
    └── Task_B/
        ├── train/
        ├── val/
        └── test/
```

```bash
cd Task-B_FM
```

#### Evaluate on TRAIN + VALIDATION

```bash
python eval_face_t_v.py
```

This will evaluate using FAISS-based cosine similarity and log metrics.

#### Evaluate on TEST

```bash
python eval_face_test.py --data_path ../Comys_Hackathon5/Task_B/test
```

#### Output Metrics

All evaluations append metrics to `metrics_output.txt`. Example:

```
# [ eval_face_test.py | EXAMPLE SET | 2025-07-02 08:25:03.012664 ]
Top-1 Accuracy      : 0.8839
Macro-averaged F1   : 0.8832
Precision           : 0.8823
Recall              : 0.8858
Threshold Used      : 0.80
```

---

### Notes

- Always run scripts inside the respective folders.
- Metrics result append to `metrics_output.txt`.
- `training_stats.txt` shows epoch-wise logs.
- `model_diagram.png` shows architecture flow.

---

## Project Structure

```
.
├──Task-A_GC/
│  ├── eval_gender_t_v.py
│  ├── eval_gender_test.py
│  ├── GCM_ep4_f1_0.9577.pth
│  ├── metrics_output.txt
│  ├── model_diagram.png
│  ├── train_gender.py
│  └── training_stats.txt
│  
├──Task-B_FM/
│  ├── eval_face_t_v.py
│  ├── eval_face_test.py
│  ├── FMM_ep3_f1_0.8832.pth
│  ├── metrics_output.txt
│  ├── model_diagram.png
│  ├── train_face.py
│  └── training_stats.txt
│  
├── check_cuda.py
├── requirements.txt
├── README.md
└── Summary.pdf
```

---

## Environment Initialization

```bash
# Clone the repository
git clone https://github.com/shivamgupta-22/Script-Sentries-CH25.git
cd Script-Sentries-CH25

# Create a Python 3.11 environment
conda create -n comsys python=3.11
conda activate comsys
# OR using venv
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt --no-deps

# Check GPU availability
python check_cuda.py
```

---

## Usage Notes

* Maintain **consistent folder structures** for `train`, `val`, and `test` directories across both tasks.
* Always execute scripts from inside the respective folders or adjust paths accordingly.

---

## Output Files

* Evaluation results are written to `metrics_output.txt`
* Model checkpoints are saved with naming format: `<prefix>_ep<epoch>_f1_<score>.pth`

---

## Acknowledgment

We extend our sincere thanks to the organizers of **COMSYS Hackathon 5 (2025)** for facilitating an impactful platform to apply AI in real-world challenges.

---

## GitHub Workflow

* Work in feature branches and open pull requests for reviews
* Use clear, descriptive commit messages
* Keep `README.md` and `requirements.txt` up to date
* Open-source contributions and forks are welcome

---

Thank you for exploring our project. 
