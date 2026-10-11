import torch
from PIL import Image

from preprocessing import test_transform
from model import CataractModel


IMAGE_PATH = "test_image.jpg"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# Load model
model = CataractModel()

model.load_state_dict(
    torch.load(
        "models/best_model.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()


# Load image
image = Image.open(IMAGE_PATH).convert("RGB")

image = test_transform(image)

image = image.unsqueeze(0)

image = image.to(DEVICE)


# Prediction
with torch.no_grad():

    output = model(image)

    probability = torch.sigmoid(output).item()


print("\nPrediction")
print("----------------")

print(
    f"Cataract probability: {probability * 100:.2f}%"
)


if probability >= 0.5:

    print("Result: POSSIBLE CATARACT")

else:

    print("Result: NORMAL")