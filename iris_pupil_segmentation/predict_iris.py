from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# MODEL
# ============================================================

from train_iris import UNet

from iris_pupil_segmentation.config import (
    DEVICE,
    IMAGE_SIZE,
    IRIS_MODEL_DIR,
    IRIS_OUTPUT_DIR,
    PREPARED_VAL_IRIS_DIR,
)


# ============================================================
# MODEL CHECKPOINT
# ============================================================

# IMPORTANT:
# Use the cataract-domain fine-tuned iris model.
MODEL_FILENAME = "best_iris_model_cataract.pth"


# ============================================================
# PREDICTION THRESHOLD
# ============================================================

MASK_THRESHOLD = 0.5


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the cataract-domain fine-tuned iris segmentation model.
    """

    model_path = IRIS_MODEL_DIR / MODEL_FILENAME

    if not model_path.exists():
        raise FileNotFoundError(
            "\nIris model checkpoint not found.\n"
            f"Expected path:\n{model_path}\n"
        )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    model = UNet().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("\nLoaded iris model:")
    print(model_path)

    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch")
    )

    print(
        "Best Dice:",
        checkpoint.get("best_dice")
    )

    return model


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):
    """
    Apply exactly the same preprocessing used by
    the iris segmentation model.
    """

    resized = TF.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    tensor = TF.to_tensor(
        resized
    )

    tensor = TF.normalize(
        tensor,
        [
            0.485,
            0.456,
            0.406
        ],
        [
            0.229,
            0.224,
            0.225
        ]
    )

    tensor = tensor.unsqueeze(
        0
    ).to(DEVICE)

    return resized, tensor


# ============================================================
# PREDICT MASK
# ============================================================

def predict_mask(
    model,
    tensor
):
    """
    Run iris segmentation.

    Returns:
        probability_map
        binary_mask
    """

    with torch.no_grad():

        output = model(
            tensor
        )

        probability = torch.sigmoid(
            output
        )[0, 0]

        mask = (
            probability > MASK_THRESHOLD
        ).cpu().numpy()

    return (
        probability.cpu().numpy(),
        mask
    )


# ============================================================
# RESIZE MASK TO ORIGINAL IMAGE
# ============================================================

def resize_mask_to_original(
    mask,
    original_size
):
    """
    Resize predicted mask from model resolution
    back to original image resolution.
    """

    mask_image = Image.fromarray(
        (
            mask * 255
        ).astype(
            np.uint8
        )
    )

    mask_image = mask_image.resize(
        original_size,
        Image.Resampling.NEAREST
    )

    return (
        np.array(mask_image) > 127
    )


# ============================================================
# CALCULATE DICE / IOU
# ============================================================

def calculate_metrics(
    prediction,
    ground_truth
):
    """
    Calculate Dice and IoU.
    """

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    union = np.logical_or(
        prediction,
        ground_truth
    ).sum()

    dice = (
        2.0 * intersection
        /
        (
            prediction.sum()
            + ground_truth.sum()
            + 1e-8
        )
    )

    iou = (
        intersection
        /
        (
            union + 1e-8
        )
    )

    return dice, iou


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(
    image_path
):
    """
    Load validation ground-truth iris mask if available.

    This is only used for validation/diagnostic mode.
    Product inference does not require a ground-truth mask.
    """

    gt_path = (
        PREPARED_VAL_IRIS_DIR
        / image_path.name
    )

    if not gt_path.exists():
        return None

    ground_truth = np.array(
        Image.open(
            gt_path
        ).convert("L")
    ) > 127

    return ground_truth


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict(
    image_path,
    save_visualization=True
):
    """
    Run iris segmentation on one image.

    Returns:
        dictionary containing:
            mask
            probability
            original_image
    """

    image_path = Path(
        image_path
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"\nInput image not found:\n{image_path}"
        )

    print("\n" + "=" * 70)
    print("TRINAY - IRIS SEGMENTATION")
    print("=" * 70)

    print("\nImage:")
    print(image_path)

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    original = Image.open(
        image_path
    ).convert("RGB")

    print(
        "Original size:",
        original.size
    )

    # ========================================================
    # PREPROCESS
    # ========================================================

    resized, tensor = preprocess_image(
        original
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    probability, mask = predict_mask(
        model,
        tensor
    )

    print(
        "Model input size:",
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    print(
        "Predicted iris pixels:",
        int(mask.sum())
    )

    print(
        "Probability min:",
        float(probability.min())
    )

    print(
        "Probability max:",
        float(probability.max())
    )

    # ========================================================
    # RESIZE MASK TO ORIGINAL IMAGE
    # ========================================================

    original_mask = resize_mask_to_original(
        mask,
        original.size
    )

    print(
        "Output mask size:",
        original_mask.shape
    )

    print(
        "Original-resolution iris pixels:",
        int(original_mask.sum())
    )

    # ========================================================
    # BASIC VALIDITY CHECK
    # ========================================================

    iris_pixels = int(
        original_mask.sum()
    )

    iris_detected = iris_pixels > 0

    print(
        "Iris detected:",
        iris_detected
    )

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    ground_truth = load_ground_truth(
        image_path
    )

    # ========================================================
    # VALIDATION MODE
    # ========================================================

    dice = None
    iou = None

    if ground_truth is not None:

        if ground_truth.shape != original_mask.shape:

            gt_image = Image.fromarray(
                (
                    ground_truth * 255
                ).astype(
                    np.uint8
                )
            )

            gt_image = gt_image.resize(
                original.size,
                Image.Resampling.NEAREST
            )

            ground_truth = (
                np.array(gt_image)
                > 127
            )

        dice, iou = calculate_metrics(
            original_mask,
            ground_truth
        )

        print(
            f"\nDice: {dice:.4f}"
        )

        print(
            f"IoU:  {iou:.4f}"
        )

    else:

        print(
            "\nNo ground-truth mask found."
        )

        print(
            "Running in PRODUCT INFERENCE mode."
        )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    if save_visualization:

        original_array = np.array(
            original
        )

        if ground_truth is not None:

            fig, axes = plt.subplots(
                1,
                5,
                figsize=(22, 5)
            )

            # ------------------------------------------------
            # 1. Original
            # ------------------------------------------------

            axes[0].imshow(
                original_array
            )

            axes[0].set_title(
                "Original Image"
            )

            axes[0].axis("off")

            # ------------------------------------------------
            # 2. Ground Truth
            # ------------------------------------------------

            axes[1].imshow(
                ground_truth,
                cmap="gray"
            )

            axes[1].set_title(
                "GT Iris"
            )

            axes[1].axis("off")

            # ------------------------------------------------
            # 3. Prediction
            # ------------------------------------------------

            axes[2].imshow(
                original_mask,
                cmap="gray"
            )

            axes[2].set_title(
                "Predicted Iris"
            )

            axes[2].axis("off")

            # ------------------------------------------------
            # 4. GT Overlay
            # ------------------------------------------------

            axes[3].imshow(
                original_array
            )

            axes[3].imshow(
                ground_truth,
                alpha=0.45
            )

            axes[3].set_title(
                "GT Overlay"
            )

            axes[3].axis("off")

            # ------------------------------------------------
            # 5. Prediction Overlay
            # ------------------------------------------------

            axes[4].imshow(
                original_array
            )

            axes[4].imshow(
                original_mask,
                alpha=0.45
            )

            axes[4].set_title(
                "Prediction Overlay"
            )

            axes[4].axis("off")

        else:

            fig, axes = plt.subplots(
                1,
                3,
                figsize=(16, 5)
            )

            # ------------------------------------------------
            # 1. Original
            # ------------------------------------------------

            axes[0].imshow(
                original_array
            )

            axes[0].set_title(
                "Original Image"
            )

            axes[0].axis("off")

            # ------------------------------------------------
            # 2. Predicted Iris
            # ------------------------------------------------

            axes[1].imshow(
                original_mask,
                cmap="gray"
            )

            axes[1].set_title(
                "Predicted Iris"
            )

            axes[1].axis("off")

            # ------------------------------------------------
            # 3. Overlay
            # ------------------------------------------------

            axes[2].imshow(
                original_array
            )

            axes[2].imshow(
                original_mask,
                alpha=0.45
            )

            axes[2].set_title(
                "Iris Overlay"
            )

            axes[2].axis("off")

        plt.tight_layout()

        output_path = (
            IRIS_OUTPUT_DIR
            / "iris_prediction.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "\nSaved:"
        )

        print(
            output_path
        )

    print("\n" + "=" * 70)
    print("IRIS PREDICTION COMPLETE")
    print("=" * 70)

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "mask": original_mask,
        "probability": probability,
        "iris_detected": iris_detected,
        "dice": dice,
        "iou": iou,
        "image": original,
    }


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            'python iris_pupil_segmentation/predict_iris.py '
            '"path/to/image.png"'
        )

        sys.exit(1)

    predict(
        sys.argv[1]
    )