from fastapi import FastAPI, UploadFile, File
from app.predict import predict

app = FastAPI()

@app.get("/")
def home():
    return {"status": "FoodVision API is running"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    label = predict(file.file)
    return {"prediction": label}
