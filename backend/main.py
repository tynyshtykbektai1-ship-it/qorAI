from fastapi import FastAPI,UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sheepcount import count_sheep_in_video
from pydantic import BaseModel
from mastitis_detection import BovineHealthAnalyzer
import io
from PIL import Image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = BovineHealthAnalyzer(
    teat_weights="models/teat_detection_enhanced.pth",
    mastitis_weights="models/mastitis_model2.pth"
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
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result2 = analyzer.predict(image)
    print(f"Diagnosis: {result2['status']}, Confidence: {result2['score']}")
    return {"mastitis_detection": result2}

