from ultralytics import YOLO
import io
from PIL import Image

class Meat_fresh_Detector:
    def __init__(self, model_path='best_meat_fresh.pt'):
        try:
            self.model = YOLO(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def predict(self, image_bytes):

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
       
        results = self.model(image)
        
        
        probs = results[0].probs
        class_id = int(probs.top1)
        confidence = round(float(probs.top1conf), 3)
        class_name = results[0].names[class_id]
        
        return class_name, confidence