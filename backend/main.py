from fastapi import FastAPI,UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sheepcount import count_sheep_in_video
from pydantic import BaseModel
from mastitis_detection import BovineHealthAnalyzer
import io
from PIL import Image
from cow_disese import Cow_disease_Detector


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer_mastitis = BovineHealthAnalyzer(
    teat_weights="models/teat_detection_enhanced.pth",
    mastitis_weights="models/mastitis_model2.pth"
)

analyzer_cow_disease = Cow_disease_Detector(
    model_path='models/best_cow-disease.pt'
)

class SensorData(BaseModel):
    Temperature: float
    Pressure: float
    CH4: float
    CO2: float
    CO: float
    NH3: float


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/sheepcount")
async def count_sheep(file: UploadFile = File(...)):


    result = count_sheep_in_video(file.file)
    return {"amount_of_ship": result}


@app.post("/sensorsdata")
async def predict(data: SensorData):
    return {"sensordata": data}

@app.post("/mastitis_detection")
async def predict_mastitis(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result2 = analyzer_mastitis.predict(image)
    print(f"Diagnosis: {result2['status']}, Confidence: {result2['score']}")
    return {"mastitis_detection": result2["status"], "confidence": result2["score"]}

@app.post('/cow_disease_detection')
async def predict_cow_disease(file: UploadFile = File(...)):
    image_bytes = await file.read()
    class_name, confidence = analyzer_cow_disease.predict(image_bytes)
    print(f"Disease: {class_name}, Confidence: {confidence}")
    return {"cow_disease": class_name, "confidence": confidence}

