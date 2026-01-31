import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

class EnhancedTeatDetector(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0'):
        super(EnhancedTeatDetector, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2) 
        )
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class BovineHealthAnalyzer:
    def __init__(self, teat_weights, mastitis_weights, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Инициализация модели сосков
        self.teat_model = EnhancedTeatDetector().to(self.device)
        t_checkpoint = torch.load(teat_weights, map_location=self.device)
        self.teat_model.load_state_dict(t_checkpoint['model_state_dict'])
        self.teat_model.eval()

        self.mastitis_model = CNNModel().to(self.device)
        self.mastitis_model.load_state_dict(torch.load(mastitis_weights, map_location=self.device))
        self.mastitis_model.eval()

        self.teat_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mastitis_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def predict(self, pil_image):
        t_input = self.teat_tf(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            t_out = self.teat_model(t_input)
            is_teat = torch.argmax(t_out, dim=1).item() == 1
        
        if not is_teat:
            return {"is_valid": False, "status": "Not a teat image", "score": 0.0}

        m_input = self.mastitis_tf(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            m_out = self.mastitis_model(m_input)
            prob = torch.sigmoid(m_out).item()
        
        return {
            "is_valid": True,
            "status": "Mastitis Detected" if prob >= 0.5 else "Healthy",
            "score": round(prob, 3),
            "severity": "High" if prob > 0.75 else "Moderate" if prob > 0.5 else "None"
        }