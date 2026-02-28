import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import os

MODEL_PATH = "solar_cnn.pth"

class SolarVision:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(MODEL_PATH)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_model(self, path):
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device).eval()
        return model

    def analyze_url(self, url):
        try:
            resp = requests.get(url, timeout=10)
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(tensor)
                prob = torch.sigmoid(output).item()
            return prob * 100, img
        except Exception:
            return 0.0, None