from ultralytics import YOLO
import io
from PIL import Image

class Cow_disease_Detector:
    def __init__(self, model_path='best_cow-disease.pt'):
        try:
            self.model = YOLO(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def predict(self, image_bytes):
        # Обработка байтов в PIL изображение
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Запуск детекции
        results = self.model(image)
        
        # Извлечение данных (как в твоем коде)
        probs = results[0].probs
        class_id = int(probs.top1)
        confidence = round(float(probs.top1conf), 3)
        class_name = results[0].names[class_id]
        
        return class_name, confidence