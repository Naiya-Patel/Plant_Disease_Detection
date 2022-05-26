from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PEPPER = tf.keras.models.load_model("/home/fx/Documents/Leaf-Disease-Classification/Pepper-Bell/Models/3")
CLASS_NAMES_PEPPER = ['Pepper-Bell Bacterial Spot', 'Healthy']

MODEL_POTATO = tf.keras.models.load_model("/home/fx/Documents/Leaf-Disease-Classification/Potato/Models/3")
CLASS_NAMES_POTATO = ['Potato Early Blight', 'Potato Late Blight', 'Healthy']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions_pe = MODEL_PEPPER.predict(img_batch)
    predictions_po = MODEL_POTATO.predict(img_batch)

    predicted_class_pe = CLASS_NAMES_PEPPER[np.argmax(predictions_pe[0])]
    confidence_pe = np.max(predictions_pe[0])

    predicted_class_po = CLASS_NAMES_POTATO[np.argmax(predictions_po[0])]
    confidence_po = np.max(predictions_po[0])

    if confidence_pe > confidence_po:
        return {
            'class': predicted_class_pe,
            'confidence': float(confidence_pe)
        }
    
    else:
        return {
            'class': predicted_class_po,
            'confidence': float(confidence_po)
        }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
