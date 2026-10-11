from pathlib import Path
import random

import pandas as pd
import numpy as np

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import matplotlib.pyplot as plt
from tqdm import tqdm

from quality_model import QualityModel


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "data/quality_labels.csv"

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs/quality")

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "quality_model.pth"

IMAGE_SIZE = 224

# Reduced from 64 for safer MPS training
BATCH_SIZE = 32

EPOCHS = 15

LEARNING_RATE = 1e-4

PATIENCE = 4

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


print("=" * 60)
print("IMAGE QUALITY MODEL TRAINING")
print("=" * 60)

print("\nDevice:", DEVICE)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)


# ============================================================
# DATASET
# ============================================================

class QualityDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.df = dataframe.reset_index(drop=True)

        self.transform = transform


    def __len__(self):

        return len(self.df)


    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = row["image_path"]

        try:

            image = Image.open(image_path).convert("RGB")

        except Exception as e:

            raise RuntimeError(
                f"\nCould not load image:\n"
                f"{image_path}\n"
                f"Error: {e}"
            )


        if self.transform:

            image = self.transform(image)


        # ----------------------------------------------------
        # LABEL
        #
        # GOOD = 1
        # BAD  = 0
        # ----------------------------------------------------

        label = 1.0 if row["quality"] == "GOOD" else 0.0


        return (
            image,
            torch.tensor(
                label,
                dtype=torch.float32
            )
        )


# ============================================================
# PRETRAINED MODEL NORMALIZATION
# ============================================================

weights = MobileNet_V3_Small_Weights.DEFAULT

mean = weights.transforms().mean

std = weights.transforms().std


# ============================================================
# TRAIN TRANSFORM
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=5
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=mean,
        std=std
    )
])


# ============================================================
# VALIDATION / TEST TRANSFORM
# ============================================================

val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=mean,
        std=std
    )
])


# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading quality labels...")

df = pd.read_csv(CSV_PATH)

print(
    f"Total images: {len(df)}"
)

print("\nOverall distribution:")

print(
    df["quality"].value_counts()
)


# ============================================================
# SPLIT DATA
# ============================================================

train_df = df[
    df["split"] == "train"
].copy()

val_df = df[
    df["split"] == "val"
].copy()

test_df = df[
    df["split"] == "test"
].copy()


print("\nDataset splits:")

print(
    f"Train      : {len(train_df)}"
)

print(
    f"Validation : {len(val_df)}"
)

print(
    f"Test       : {len(test_df)}"
)


# ============================================================
# CHECK SPLIT DISTRIBUTION
# ============================================================

print("\nTrain distribution:")

print(
    train_df["quality"].value_counts()
)


print("\nValidation distribution:")

print(
    val_df["quality"].value_counts()
)


print("\nTest distribution:")

print(
    test_df["quality"].value_counts()
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = QualityDataset(
    train_df,
    transform=train_transform
)


val_dataset = QualityDataset(
    val_df,
    transform=val_transform
)


# ============================================================
# DATALOADERS
# ============================================================
#
# num_workers=0 is intentional for macOS/MPS reliability.
#
# Once training is confirmed working, we can benchmark
# num_workers=2 or 4 if necessary.
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


# ============================================================
# DATA LOADING TEST
# ============================================================

print("\n" + "=" * 60)
print("TESTING DATA LOADING")
print("=" * 60)

print("\nLoading one training batch...")

test_images, test_labels = next(
    iter(train_loader)
)

print(
    "Batch images:",
    test_images.shape
)

print(
    "Batch labels:",
    test_labels.shape
)

print(
    "Example labels:",
    test_labels[:10].tolist()
)

print(
    "Image tensor dtype:",
    test_images.dtype
)

print(
    "Data loading test successful."
)


# ============================================================
# MODEL
# ============================================================

print("\n" + "=" * 60)
print("CREATING MODEL")
print("=" * 60)

model = QualityModel()

model = model.to(DEVICE)

print("\nMobileNetV3-Small loaded.")

print(
    "Model device:",
    DEVICE
)


# ============================================================
# CLASS WEIGHT
# ============================================================

good_count = (
    train_df["quality"] == "GOOD"
).sum()


bad_count = (
    train_df["quality"] == "BAD"
).sum()


# GOOD = 1
# BAD  = 0
#
# GOOD is the minority class.
#
# This makes the positive GOOD class more important
# during training.

pos_weight_value = (
    bad_count / good_count
)


pos_weight = torch.tensor(
    [pos_weight_value],
    dtype=torch.float32,
    device=DEVICE
)


print("\n" + "=" * 60)
print("CLASS BALANCING")
print("=" * 60)

print(
    "GOOD:",
    good_count
)

print(
    "BAD :",
    bad_count
)

print(
    "Positive class weight:",
    pos_weight.item()
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# TRAINING HISTORY
# ============================================================

train_losses = []

val_losses = []

val_accuracies = []

val_precisions = []

val_recalls = []

val_specificities = []

val_f1s = []

val_aucs = []


# ============================================================
# BEST MODEL TRACKING
# ============================================================

best_auc = 0.0

patience_counter = 0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):

    print("\n")

    print("=" * 70)

    print(
        f"EPOCH {epoch + 1}/{EPOCHS}"
    )

    print("=" * 70)


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_loss = 0.0


    train_progress = tqdm(
        train_loader,
        desc=f"Training Epoch {epoch + 1}",
        total=len(train_loader),
        unit="batch"
    )


    for batch_index, (images, labels) in enumerate(
        train_progress
    ):

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        # ----------------------------------------------------
        # CLEAR GRADIENTS
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        outputs = model(
            images
        ).squeeze(1)


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels
        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # UPDATE
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # TRACK LOSS
        # ----------------------------------------------------

        running_loss += (
            loss.item() *
            images.size(0)
        )


        # ----------------------------------------------------
        # UPDATE PROGRESS BAR
        # ----------------------------------------------------

        train_progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    train_loss = (
        running_loss /
        len(train_dataset)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0

    all_labels = []

    all_probs = []


    val_progress = tqdm(
        val_loader,
        desc="Validation",
        total=len(val_loader),
        unit="batch"
    )


    with torch.no_grad():

        for images, labels in val_progress:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            outputs = model(
                images
            ).squeeze(1)


            loss = criterion(
                outputs,
                labels
            )


            val_loss += (
                loss.item() *
                images.size(0)
            )


            probs = torch.sigmoid(
                outputs
            )


            all_probs.extend(
                probs.detach()
                .cpu()
                .numpy()
            )


            all_labels.extend(
                labels.cpu()
                .numpy()
            )


    val_loss /= len(
        val_dataset
    )


    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    all_labels = np.array(
        all_labels
    )

    all_probs = np.array(
        all_probs
    )


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    predictions = (
        all_probs >= 0.5
    ).astype(int)


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        all_labels,
        predictions
    )


    precision = precision_score(
        all_labels,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        all_labels,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        all_labels,
        predictions,
        zero_division=0
    )


    auc = roc_auc_score(
        all_labels,
        all_probs
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        all_labels,
        predictions,
        labels=[0, 1]
    )


    tn, fp, fn, tp = cm.ravel()


    # ========================================================
    # SPECIFICITY
    # ========================================================

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    val_accuracies.append(
        accuracy
    )

    val_precisions.append(
        precision
    )

    val_recalls.append(
        recall
    )

    val_specificities.append(
        specificity
    )

    val_f1s.append(
        f1
    )

    val_aucs.append(
        auc
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")

    print(
        f"Train Loss   : {train_loss:.4f}"
    )

    print(
        f"Val Loss     : {val_loss:.4f}"
    )

    print(
        f"Accuracy     : {accuracy:.4f}"
    )

    print(
        f"Precision    : {precision:.4f}"
    )

    print(
        f"GOOD Recall  : {recall:.4f}"
    )

    print(
        f"BAD Specificity: {specificity:.4f}"
    )

    print(
        f"F1 Score     : {f1:.4f}"
    )

    print(
        f"ROC-AUC      : {auc:.4f}"
    )


    print("\nConfusion Matrix:")

    print(cm)


    # ========================================================
    # LEARNING RATE
    # ========================================================

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"\nLearning Rate: {current_lr:.7f}"
    )


    scheduler.step(
        auc
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if auc > best_auc:

        best_auc = auc

        patience_counter = 0


        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


        print(
            "\n✓ BEST MODEL SAVED"
        )

        print(
            f"ROC-AUC: {best_auc:.4f}"
        )

        print(
            f"Path: {MODEL_PATH}"
        )


    else:

        patience_counter += 1


        print(
            f"\nNo improvement "
            f"({patience_counter}/{PATIENCE})"
        )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if patience_counter >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n")

print("=" * 70)

print("TRAINING COMPLETE")

print("=" * 70)

print(
    f"\nBest validation ROC-AUC: "
    f"{best_auc:.4f}"
)

print(
    f"Model saved at:"
)

print(
    MODEL_PATH
)


# ============================================================
# SAVE TRAINING CURVE
# ============================================================

epochs_completed = range(
    1,
    len(train_losses) + 1
)


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs_completed,
    train_losses,
    label="Training Loss"
)

plt.plot(
    epochs_completed,
    val_losses,
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Image Quality Model - Training and Validation Loss"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


loss_curve_path = (
    OUTPUT_DIR /
    "training_loss_curve.png"
)


plt.savefig(
    loss_curve_path,
    dpi=150
)

plt.close()


# ============================================================
# AUC CURVE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs_completed,
    val_aucs,
    marker="o",
    label="Validation ROC-AUC"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "ROC-AUC"
)

plt.title(
    "Image Quality Model - Validation ROC-AUC"
)

plt.ylim(
    0.0,
    1.05
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


auc_curve_path = (
    OUTPUT_DIR /
    "validation_auc_curve.png"
)


plt.savefig(
    auc_curve_path,
    dpi=150
)

plt.close()


# ============================================================
# SAVE METRICS CSV
# ============================================================

history_df = pd.DataFrame({

    "epoch": list(
        epochs_completed
    ),

    "train_loss": train_losses,

    "val_loss": val_losses,

    "accuracy": val_accuracies,

    "precision": val_precisions,

    "good_recall": val_recalls,

    "bad_specificity": val_specificities,

    "f1": val_f1s,

    "roc_auc": val_aucs

})


history_path = (
    OUTPUT_DIR /
    "training_history.csv"
)


history_df.to_csv(
    history_path,
    index=False
)


print("\nOutput files:")

print(
    f"Loss curve: {loss_curve_path}"
)

print(
    f"AUC curve : {auc_curve_path}"
)

print(
    f"History   : {history_path}"
)

print(
    f"Model     : {MODEL_PATH}"
)

print("\nDone.")