import tensorflow as tf
from PIL import Image
import numpy as np
import io

class MilkYieldPredictor:
    def __init__(self, model_path='models/milk_yield_model.h5'):
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def preprocess_img(self, file_bytes, target_size=(128, 128)):
        img = Image.open(io.BytesIO(file_bytes))
        img = img.resize(target_size)
        img = img.convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict_milk_yield(self, file_bytes):
        try:
            img_array = self.preprocess_img(file_bytes)
            prediction = self.model.predict(img_array)
            predicted_yield = prediction[0][0]
            return {"predicted_milk_yield": round(float(predicted_yield), 3)}
        except Exception as e:
            return {"error": str(e)}
