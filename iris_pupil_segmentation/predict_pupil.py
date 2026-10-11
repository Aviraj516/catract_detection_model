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

from train_pupil import UNet

from iris_pupil_segmentation.config import (
    DEVICE,
    IMAGE_SIZE,
    PUPIL_MODEL_DIR,
    PUPIL_OUTPUT_DIR,
    PREPARED_VAL_PUPIL_DIR,
)


# ============================================================
# MODEL CHECKPOINT
# ============================================================

# Cataract-1K fine-tuned pupil model
MODEL_FILENAME = "best_pupil_model_cataract1k.pth"


# ============================================================
# SEGMENTATION THRESHOLD
# ============================================================

MASK_THRESHOLD = 0.5


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the cataract-domain fine-tuned pupil model.

    This function is intentionally separate so the master
    TRINAY pipeline can load the model once and reuse it.
    """

    model_path = (
        PUPIL_MODEL_DIR
        / MODEL_FILENAME
    )

    if not model_path.exists():

        raise FileNotFoundError(
            "\nPupil model checkpoint not found.\n"
            f"Expected:\n{model_path}\n"
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

    print("\nLoaded pupil model:")
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
    Apply the same preprocessing used during pupil training.

    Input:
        PIL RGB image

    Output:
        resized PIL image
        normalized tensor
    """

    resized = TF.resize(
        image,
        [
            IMAGE_SIZE,
            IMAGE_SIZE
        ]
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
    Run pupil segmentation.

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
    Convert the 256x256 prediction back to the original
    Eye ROI resolution.

    This is important for the integrated pipeline because
    the pupil mask must align pixel-for-pixel with the
    original Eye ROI.
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
    Load validation ground-truth pupil mask if available.

    Product inference does not require ground truth.
    """

    gt_path = (
        PREPARED_VAL_PUPIL_DIR
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
    model=None,
    save_visualization=True
):
    """
    Run pupil segmentation.

    Parameters
    ----------
    image_path:
        Path to an image / Eye ROI.

    model:
        Optional already-loaded pupil model.

        The master TRINAY pipeline should pass the loaded
        model here so the checkpoint is not loaded for
        every image.

    save_visualization:
        Save diagnostic visualization when True.

    Returns
    -------
    dict
        Contains the original-resolution pupil mask and
        prediction information.
    """

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"\nInput image not found:\n{image_path}"
        )

    print("\n" + "=" * 70)
    print("TRINAY - PUPIL SEGMENTATION")
    print("=" * 70)

    print("\nImage:")
    print(image_path)

    # ========================================================
    # LOAD MODEL ONLY IF NOT PROVIDED
    # ========================================================

    if model is None:

        model = load_model()

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    original = Image.open(
        image_path
    ).convert("RGB")

    original_size = original.size

    print(
        "Original size:",
        original_size
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
        "Predicted pupil pixels:",
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
    # RESIZE MASK TO ORIGINAL ROI
    # ========================================================

    original_mask = resize_mask_to_original(
        mask,
        original_size
    )

    print(
        "Output mask size:",
        original_mask.shape
    )

    print(
        "Original-resolution pupil pixels:",
        int(original_mask.sum())
    )

    # ========================================================
    # PUPIL DETECTION CHECK
    # ========================================================

    pupil_pixels = int(
        original_mask.sum()
    )

    pupil_detected = (
        pupil_pixels > 0
    )

    print(
        "Pupil detected:",
        pupil_detected
    )

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    ground_truth = load_ground_truth(
        image_path
    )

    dice = None
    iou = None

    ground_truth_resized = None

    # ========================================================
    # VALIDATION MODE
    # ========================================================

    if ground_truth is not None:

        # Ground truth is normally at the original resolution.
        # Resize it to the original image size if required.

        if ground_truth.shape != original_mask.shape:

            gt_image = Image.fromarray(
                (
                    ground_truth * 255
                ).astype(
                    np.uint8
                )
            )

            gt_image = gt_image.resize(
                original_size,
                Image.Resampling.NEAREST
            )

            ground_truth = (
                np.array(gt_image)
                > 127
            )

        # Resize GT to model resolution for comparison
        # with the raw model prediction.

        gt_image_model = Image.fromarray(
            (
                ground_truth * 255
            ).astype(
                np.uint8
            )
        )

        gt_image_model = TF.resize(
            gt_image_model,
            [
                IMAGE_SIZE,
                IMAGE_SIZE
            ],
            interpolation=TF.InterpolationMode.NEAREST
        )

        ground_truth_resized = (
            np.array(gt_image_model)
            > 127
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        dice, iou = calculate_metrics(
            mask,
            ground_truth_resized
        )

        print(
            f"\nDice: {dice:.4f}"
        )

        print(
            f"IoU:  {iou:.4f}"
        )

    else:

        print(
            "\nNo ground truth found."
        )

        print(
            "Running in PRODUCT INFERENCE mode."
        )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    if save_visualization:

        image_array = np.array(
            resized
        )

        if ground_truth_resized is not None:

            fig, axes = plt.subplots(
                1,
                5,
                figsize=(22, 5)
            )

            # ------------------------------------------------
            # 1. Image
            # ------------------------------------------------

            axes[0].imshow(
                image_array
            )

            axes[0].set_title(
                "Image"
            )

            axes[0].axis("off")

            # ------------------------------------------------
            # 2. Ground Truth
            # ------------------------------------------------

            axes[1].imshow(
                ground_truth_resized,
                cmap="gray"
            )

            axes[1].set_title(
                "GT Pupil"
            )

            axes[1].axis("off")

            # ------------------------------------------------
            # 3. Prediction
            # ------------------------------------------------

            axes[2].imshow(
                mask,
                cmap="gray"
            )

            axes[2].set_title(
                "Predicted Pupil"
            )

            axes[2].axis("off")

            # ------------------------------------------------
            # 4. GT Overlay
            # ------------------------------------------------

            axes[3].imshow(
                image_array
            )

            axes[3].imshow(
                ground_truth_resized,
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
                image_array
            )

            axes[4].imshow(
                mask,
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
                figsize=(15, 5)
            )

            # ------------------------------------------------
            # 1. Image
            # ------------------------------------------------

            axes[0].imshow(
                image_array
            )

            axes[0].set_title(
                "Image"
            )

            axes[0].axis("off")

            # ------------------------------------------------
            # 2. Prediction
            # ------------------------------------------------

            axes[1].imshow(
                mask,
                cmap="gray"
            )

            axes[1].set_title(
                "Predicted Pupil"
            )

            axes[1].axis("off")

            # ------------------------------------------------
            # 3. Prediction Overlay
            # ------------------------------------------------

            axes[2].imshow(
                image_array
            )

            axes[2].imshow(
                mask,
                alpha=0.45
            )

            axes[2].set_title(
                "Prediction Overlay"
            )

            axes[2].axis("off")

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        plt.tight_layout()

        output_path = (
            PUPIL_OUTPUT_DIR
            / "pupil_cataract1k_diagnostic.png"
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

    # ========================================================
    # RETURN PIPELINE RESULT
    # ========================================================

    return {
        "mask": original_mask,
        "probability": probability,
        "pupil_detected": pupil_detected,
        "pupil_pixels": pupil_pixels,
        "dice": dice,
        "iou": iou,
        "image": original,
    }


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            'python3 iris_pupil_segmentation/predict_pupil.py '
            '"path/to/image.png"'
        )

        sys.exit(1)

    predict(
        sys.argv[1]
    )