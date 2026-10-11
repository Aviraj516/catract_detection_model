from pathlib import Path
import torch
from ultralytics import YOLO


# ============================================================
# PROJECT ROOT
# ============================================================
# Run this script from the cataract_model directory.
PROJECT_ROOT = Path.cwd()


# ============================================================
# PATHS
# ============================================================

DATA_YAML = PROJECT_ROOT / "data" / "eye_detection" / "data.yaml"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eye_detection"


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Apple Silicon MPS detected.")
else:
    DEVICE = "cpu"
    print("MPS not available. Using CPU.")


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("TRINAY - EYE DETECTION TRAINING")
print("=" * 70)

print(f"\nProject root:")
print(PROJECT_ROOT)

print(f"\nDataset YAML:")
print(DATA_YAML)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print(f"\nDevice:")
print(DEVICE)


# ============================================================
# CHECK DATASET YAML
# ============================================================

if not DATA_YAML.exists():
    print("\nERROR:")
    print("data.yaml was not found.")
    print(f"\nExpected location:")
    print(DATA_YAML)
    raise SystemExit(1)


print("\nDataset YAML found successfully.")


# ============================================================
# CHECK DATASET DIRECTORIES
# ============================================================

TRAIN_IMAGES = PROJECT_ROOT / "data" / "eye_detection" / "train" / "images"
TRAIN_LABELS = PROJECT_ROOT / "data" / "eye_detection" / "train" / "labels"

VAL_IMAGES = PROJECT_ROOT / "data" / "eye_detection" / "val" / "images"
VAL_LABELS = PROJECT_ROOT / "data" / "eye_detection" / "val" / "labels"

TEST_IMAGES = PROJECT_ROOT / "data" / "eye_detection" / "test" / "images"
TEST_LABELS = PROJECT_ROOT / "data" / "eye_detection" / "test" / "labels"


required_paths = [
    TRAIN_IMAGES,
    TRAIN_LABELS,
    VAL_IMAGES,
    VAL_LABELS,
    TEST_IMAGES,
    TEST_LABELS,
]


for path in required_paths:
    if not path.exists():
        print(f"\nERROR: Missing dataset path:")
        print(path)
        raise SystemExit(1)


print("\nDataset folders verified successfully.")


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("\nLoading YOLO11n pretrained model...")

model = YOLO("yolo11n.pt")


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...\n")

results = model.train(
    data=str(DATA_YAML),

    # Training
    epochs=20,
    imgsz=640,
    batch=16,

    # Device
    device=DEVICE,

    # Data loading
    workers=2,

    # Early stopping
    patience=10,

    # Pretrained model
    pretrained=True,

    # Output
    project=str(OUTPUT_DIR),
    name="yolo11n_eye",

    # Saving
    save=True,
    save_period=5,

    # Validation
    val=True,

    # Do not cache dataset
    cache=False,

    # Basic augmentation
    degrees=5,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,

    # Output information
    verbose=True,
)


# ============================================================
# TRAINING COMPLETE
# ============================================================

BEST_MODEL = (
    OUTPUT_DIR
    / "yolo11n_eye"
    / "weights"
    / "best.pt"
)

LAST_MODEL = (
    OUTPUT_DIR
    / "yolo11n_eye"
    / "weights"
    / "last.pt"
)


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\nBest model:")
print(BEST_MODEL)

print("\nLast model:")
print(LAST_MODEL)

print("\nTraining results:")
print(OUTPUT_DIR / "yolo11n_eye")