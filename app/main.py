from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from PIL import Image
import numpy as np
import io
import os
import gdown

app = FastAPI(title="Eye Disease Classification API")

# Caminho do modelo
MODEL_PATH = "best_model.keras"
# URL do Google Drive (ID direto)
MODEL_URL = "https://drive.google.com/uc?id=1vSIfD3viT5JSxpG4asA8APCwK0JK9Dvu"  # <-- Substitua pelo ID correto

# Baixar o modelo se ele não existir
if not os.path.exists(MODEL_PATH):
    print("🔽 Baixando o modelo...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Carregar o modelo
print("✅ Carregando modelo...")
model = load_model(MODEL_PATH)

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
