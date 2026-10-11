import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from tqdm import tqdm

from preprocessing import train_transform, test_transform
from model import CataractModel


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/processed_images"

BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4

VAL_SIZE = 0.20
RANDOM_SEED = 42

PATIENCE = 4


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

elif torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

else:

    DEVICE = torch.device("cpu")


print("Using device:", DEVICE)


# ============================================================
# DATASET
# ============================================================

train_path = os.path.join(DATA_DIR, "train")
test_path = os.path.join(DATA_DIR, "test")


# Dataset for obtaining labels
full_dataset = ImageFolder(
    train_path
)


print("\nClasses:")
print(full_dataset.class_to_idx)

print("\nTotal training images:", len(full_dataset))


# ============================================================
# CREATE STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

indices = np.arange(len(full_dataset))
labels = np.array(full_dataset.targets)


train_indices, val_indices = train_test_split(
    indices,
    test_size=VAL_SIZE,
    random_state=RANDOM_SEED,
    stratify=labels
)


print("\nDataset split:")
print("Training:", len(train_indices))
print("Validation:", len(val_indices))


# ============================================================
# CREATE DATASETS WITH DIFFERENT TRANSFORMS
# ============================================================

train_dataset_full = ImageFolder(
    train_path,
    transform=train_transform
)

val_dataset_full = ImageFolder(
    train_path,
    transform=test_transform
)


train_dataset = Subset(
    train_dataset_full,
    train_indices
)

val_dataset = Subset(
    val_dataset_full,
    val_indices
)


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = ImageFolder(
    test_path,
    transform=test_transform
)


print("Testing:", len(test_dataset))


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = CataractModel()

model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss()


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
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# STORAGE FOR GRAPHS
# ============================================================

train_losses = []
val_losses = []

val_accuracies = []
val_sensitivities = []
val_specificities = []
val_f1_scores = []
val_aucs = []


# ============================================================
# BEST MODEL
# ============================================================

best_val_loss = float("inf")

patience_counter = 0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):

    print("\n" + "=" * 60)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print("=" * 60)


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_train_loss = 0.0


    progress_bar = tqdm(
        train_loader,
        desc="Training"
    )


    for images, labels_batch in progress_bar:

        images = images.to(DEVICE)

        # ImageFolder:
        # cataract = 0
        # normal   = 1
        #
        # Convert:
        # cataract = 1
        # normal   = 0

        labels_batch = (
            labels_batch == 0
        ).float().to(DEVICE)


        optimizer.zero_grad()


        outputs = model(images)

        outputs = outputs.squeeze(1)


        loss = criterion(
            outputs,
            labels_batch
        )


        loss.backward()

        optimizer.step()


        running_train_loss += loss.item()


        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    train_loss = (
        running_train_loss /
        len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    running_val_loss = 0.0

    all_labels = []
    all_probabilities = []


    with torch.no_grad():

        for images, labels_batch in val_loader:

            images = images.to(DEVICE)

            labels_batch = (
                labels_batch == 0
            ).float().to(DEVICE)


            outputs = model(images)

            outputs = outputs.squeeze(1)


            loss = criterion(
                outputs,
                labels_batch
            )


            running_val_loss += loss.item()


            probabilities = torch.sigmoid(
                outputs
            )


            all_labels.extend(
                labels_batch.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )


    val_loss = (
        running_val_loss /
        len(val_loader)
    )


    all_labels = np.array(all_labels)

    all_probabilities = np.array(
        all_probabilities
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    predictions = (
        all_probabilities >= 0.5
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


    sensitivity = recall_score(
        all_labels,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        all_labels,
        predictions,
        zero_division=0
    )


    # ROC-AUC
    if len(np.unique(all_labels)) == 2:

        auc = roc_auc_score(
            all_labels,
            all_probabilities
        )

    else:

        auc = 0.0


    # ========================================================
    # SPECIFICITY
    # ========================================================

    cm = confusion_matrix(
        all_labels,
        predictions,
        labels=[0, 1]
    )


    tn, fp, fn, tp = cm.ravel()


    if (tn + fp) > 0:

        specificity = (
            tn /
            (tn + fp)
        )

    else:

        specificity = 0.0


    # ========================================================
    # STORE METRICS
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

    val_sensitivities.append(
        sensitivity
    )

    val_specificities.append(
        specificity
    )

    val_f1_scores.append(
        f1
    )

    val_aucs.append(
        auc
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        f"\nTrain Loss:       {train_loss:.4f}"
    )

    print(
        f"Validation Loss:  {val_loss:.4f}"
    )

    print(
        f"Accuracy:         {accuracy:.4f}"
    )

    print(
        f"Precision:        {precision:.4f}"
    )

    print(
        f"Sensitivity:      {sensitivity:.4f}"
    )

    print(
        f"Specificity:      {specificity:.4f}"
    )

    print(
        f"F1 Score:         {f1:.4f}"
    )

    print(
        f"ROC-AUC:          {auc:.4f}"
    )


    print("\nConfusion Matrix:")

    print(cm)


    # ========================================================
    # LR SCHEDULER
    # ========================================================

    scheduler.step(
        val_loss
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        patience_counter = 0

        os.makedirs(
            "models",
            exist_ok=True
        )


        torch.save(
            model.state_dict(),
            "models/best_model.pth"
        )


        print(
            "\n✓ Best model saved"
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

print("\n" + "=" * 60)

print("TRAINING COMPLETED")

print("=" * 60)


# ============================================================
# SAVE TRAINING CURVES
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


epochs_completed = range(
    1,
    len(train_losses) + 1
)


# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(figsize=(8, 5))

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

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training vs Validation Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/training_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# METRICS CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_completed,
    val_accuracies,
    label="Accuracy"
)

plt.plot(
    epochs_completed,
    val_sensitivities,
    label="Sensitivity"
)

plt.plot(
    epochs_completed,
    val_specificities,
    label="Specificity"
)

plt.plot(
    epochs_completed,
    val_f1_scores,
    label="F1 Score"
)

plt.plot(
    epochs_completed,
    val_aucs,
    label="ROC-AUC"
)

plt.xlabel("Epoch")

plt.ylabel("Score")

plt.title(
    "Validation Metrics"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/validation_metrics.png",
    dpi=300
)

plt.close()


print(
    "\nSaved:"
)

print(
    "models/best_model.pth"
)

print(
    "outputs/training_curve.png"
)

print(
    "outputs/validation_metrics.png"
)