# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from PIL import Image
import numpy as np
import io

app = FastAPI(title="Eye Disease Classification API")

# Carregar o modelo
model = load_model("eye_disease_model.h5")

# Nome das classes
class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = image.resize((256, 256))
        img_array = np.array(image)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        return {"prediction": predicted_class, "confidence": confidence}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
