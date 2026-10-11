from pathlib import Path
import sys

import numpy as np

import torch
import torch.nn as nn

from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision.transforms.functional as TF

from tqdm import tqdm


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

sys.path.append(
    str(PROJECT_ROOT)
)


from iris_pupil_segmentation.config import (
    PREPARED_TRAIN_IMAGE_DIR,
    PREPARED_TRAIN_PUPIL_DIR,
    PREPARED_VAL_IMAGE_DIR,
    PREPARED_VAL_PUPIL_DIR,
    PUPIL_MODEL_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    DEVICE
)


class SegmentationDataset(
    Dataset
):

    def __init__(
        self,
        image_dir,
        mask_dir
    ):

        self.image_dir = Path(
            image_dir
        )

        self.mask_dir = Path(
            mask_dir
        )

        self.images = sorted(
            [
                p
                for p in self.image_dir.iterdir()
                if p.is_file()
            ]
        )

        self.mask_lookup = {
            p.stem: p
            for p in self.mask_dir.iterdir()
            if p.is_file()
        }

        self.samples = []

        for image in self.images:

            if image.stem in self.mask_lookup:

                self.samples.append(
                    (
                        image,
                        self.mask_lookup[
                            image.stem
                        ]
                    )
                )

        if not self.samples:

            raise RuntimeError(
                "No image/mask pairs found."
            )

    def __len__(self):

        return len(self.samples)

    def __getitem__(
        self,
        index
    ):

        image_path, mask_path = (
            self.samples[index]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("L")

        image = TF.resize(
            image,
            [
                IMAGE_SIZE,
                IMAGE_SIZE
            ]
        )

        mask = TF.resize(
            mask,
            [
                IMAGE_SIZE,
                IMAGE_SIZE
            ],
            interpolation=TF.InterpolationMode.NEAREST
        )

        image = TF.to_tensor(
            image
        )

        image = TF.normalize(
            image,
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

        mask = np.array(
            mask
        )

        mask = (
            mask > 127
        ).astype(
            np.float32
        )

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        return image, mask


class DoubleConv(
    nn.Module
):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(
        self,
        x
    ):

        return self.block(x)


class UNet(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.enc1 = DoubleConv(
            3,
            64
        )

        self.enc2 = DoubleConv(
            64,
            128
        )

        self.enc3 = DoubleConv(
            128,
            256
        )

        self.enc4 = DoubleConv(
            256,
            512
        )

        self.pool = nn.MaxPool2d(
            2
        )

        self.bottleneck = DoubleConv(
            512,
            1024
        )

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            2,
            stride=2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )

        self.final = nn.Conv2d(
            64,
            1,
            1
        )

    def forward(
        self,
        x
    ):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.final(d1)


def dice_loss(
    logits,
    targets
):

    probabilities = torch.sigmoid(
        logits
    )

    probabilities = probabilities.flatten(
        1
    )

    targets = targets.flatten(
        1
    )

    intersection = (
        probabilities * targets
    ).sum(dim=1)

    dice = (
        2 * intersection + 1e-6
    ) / (
        probabilities.sum(dim=1)
        + targets.sum(dim=1)
        + 1e-6
    )

    return 1 - dice.mean()


def loss_function(
    logits,
    targets
):

    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets
    )

    dice = dice_loss(
        logits,
        targets
    )

    return bce + dice


def calculate_dice(
    logits,
    targets
):

    predictions = (
        torch.sigmoid(logits)
        > 0.5
    )

    predictions = predictions.float()

    intersection = (
        predictions * targets
    ).sum()

    denominator = (
        predictions.sum()
        + targets.sum()
    )

    return (
        2 * intersection + 1e-6
    ) / (
        denominator + 1e-6
    )


def main():

    print("=" * 70)

    print(
        "TRINAY - PUPIL SEGMENTATION"
    )

    print("=" * 70)

    print(
        "Device:",
        DEVICE
    )

    train_dataset = SegmentationDataset(
        PREPARED_TRAIN_IMAGE_DIR,
        PREPARED_TRAIN_PUPIL_DIR
    )

    val_dataset = SegmentationDataset(
        PREPARED_VAL_IMAGE_DIR,
        PREPARED_VAL_PUPIL_DIR
    )

    print(
        "Training:",
        len(train_dataset)
    )

    print(
        "Validation:",
        len(val_dataset)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = UNet().to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_dice = 0.0

    model_path = (
        PUPIL_MODEL_DIR
        / "best_pupil_model.pth"
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_loss = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}"
        )

        for images, masks in progress:

            images = images.to(
                DEVICE
            )

            masks = masks.to(
                DEVICE
            )

            optimizer.zero_grad()

            outputs = model(
                images
            )

            loss = loss_function(
                outputs,
                masks
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = (
            total_loss
            / len(train_loader)
        )

        model.eval()

        val_loss = 0

        val_dice = 0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(
                    DEVICE
                )

                masks = masks.to(
                    DEVICE
                )

                outputs = model(
                    images
                )

                loss = loss_function(
                    outputs,
                    masks
                )

                val_loss += (
                    loss.item()
                )

                val_dice += (
                    calculate_dice(
                        outputs,
                        masks
                    ).item()
                )

        val_loss /= len(
            val_loader
        )

        val_dice /= len(
            val_loader
        )

        print(
            f"\nEpoch {epoch}"
        )

        print(
            f"Train Loss : {train_loss:.4f}"
        )

        print(
            f"Val Loss   : {val_loss:.4f}"
        )

        print(
            f"Val Dice   : {val_dice:.4f}"
        )

        if val_dice > best_dice:

            best_dice = val_dice

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "best_dice":
                        best_dice,

                    "epoch":
                        epoch
                },
                model_path
            )

            print(
                "✓ Best pupil model saved"
            )

    print("\nTraining complete.")

    print(
        "Best Dice:",
        best_dice
    )

    print(
        "Saved:",
        model_path
    )


if __name__ == "__main__":

    main()