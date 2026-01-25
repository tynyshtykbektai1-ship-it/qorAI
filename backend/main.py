from fastapi import FastAPI,UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sheepcount import count_sheep_in_video
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/sheepcount")
async def count_sheep(file: UploadFile = File(...)):


    result = count_sheep_in_video
    return {"amount_of_ship": result}


@app.post("/sensorsdata")
async def predict(file: UploadFile = File(...)):
    return {"filename": file.filename}

