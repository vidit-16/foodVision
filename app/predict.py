import json
import torch
from torchvision import models
from app.utils import load_image

device = "cuda" if torch.cuda.is_available() else "cpu"

# load class names
with open("model/classes.json", "r") as f:
    CLASSES = json.load(f)

# load model
def load_model():
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load("model/food_vision_resnet50.pt", map_location=device))
    model.eval()
    model.to(device)
    return model

model = load_model()

def predict(image_file):
    image = load_image(image_file).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return CLASSES[predicted.item()]
