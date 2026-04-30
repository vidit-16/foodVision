from PIL import Image
import torch
from torchvision import transforms

def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]
    )
    return transform(image).unsqueeze(0)  # add batch dimension
