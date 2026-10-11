from pathlib import Path
import torch


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


print(
    "Using device:",
    DEVICE
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 8

EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

NUM_WORKERS = 0


# ============================================================
# IMAGE SIZE
# ============================================================

IMAGE_SIZE = 256


# ============================================================
# ORIGINAL DATASET
# ============================================================

ORIGINAL_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "IRIS + PUPIL + EYE"
)


TRAIN_IMAGE_DIR = (
    ORIGINAL_DATASET_DIR
    / "train"
    / "image"
)


TRAIN_MASK_DIR = (
    ORIGINAL_DATASET_DIR
    / "train"
    / "segmentation"
)


VAL_IMAGE_DIR = (
    ORIGINAL_DATASET_DIR
    / "val"
    / "image"
)


VAL_MASK_DIR = (
    ORIGINAL_DATASET_DIR
    / "val"
    / "segmentation"
)


# ============================================================
# PREPARED DATASET
# ============================================================

PREPARED_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "iris_pupil_prepared"
)


PREPARED_TRAIN_IMAGE_DIR = (
    PREPARED_DATASET_DIR
    / "train"
    / "image"
)


PREPARED_VAL_IMAGE_DIR = (
    PREPARED_DATASET_DIR
    / "val"
    / "image"
)


PREPARED_TRAIN_IRIS_DIR = (
    PREPARED_DATASET_DIR
    / "train"
    / "iris"
)


PREPARED_VAL_IRIS_DIR = (
    PREPARED_DATASET_DIR
    / "val"
    / "iris"
)


PREPARED_TRAIN_PUPIL_DIR = (
    PREPARED_DATASET_DIR
    / "train"
    / "pupil"
)


PREPARED_VAL_PUPIL_DIR = (
    PREPARED_DATASET_DIR
    / "val"
    / "pupil"


)


# ============================================================
# LABEL MAPPING
# ============================================================

# Original segmentation dataset:
#
# 0 = background
# 1 = eye / sclera
# 2 = pupil
# 3 = iris

BACKGROUND_LABEL = 0

EYE_LABEL = 1

PUPIL_LABEL = 2

IRIS_LABEL = 3


# ============================================================
# MODEL DIRECTORIES
# ============================================================

IRIS_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "iris_segmentation"
)


PUPIL_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "pupil_segmentation"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

IRIS_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "iris_pupil"
    / "iris"
)


PUPIL_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "iris_pupil"
    / "pupil"
)


IRIS_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


PUPIL_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)