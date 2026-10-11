from pathlib import Path
from ultralytics import YOLO


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path.cwd()

# Trained model
MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "eye_detection"
    / "yolo11n_eye"
    / "weights"
    / "best.pt"
)

# Dataset
DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "eye_detection"
    / "data.yaml"
)


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("TRINAY - EYE DETECTION MODEL EVALUATION")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nDataset:")
print(DATA_YAML)


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"\ndata.yaml not found:\n{DATA_YAML}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = YOLO(str(MODEL_PATH))


# ============================================================
# TEST SET EVALUATION
# ============================================================

print("\nEvaluating on TEST dataset...\n")

metrics = model.val(
    data=str(DATA_YAML),

    split="test",

    imgsz=640,

    batch=16,

    device="mps",

    workers=2,

    plots=True,

    save_json=True,

    verbose=True
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(f"\nBox Precision : {metrics.box.mp:.4f}")
print(f"Box Recall    : {metrics.box.mr:.4f}")
print(f"mAP50         : {metrics.box.map50:.4f}")
print(f"mAP50-95      : {metrics.box.map:.4f}")

print("\nEvaluation complete.")