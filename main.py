from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
import gdown
import os
from io import BytesIO

app = Flask(__name__)

MODEL_PATH = "model/best_model.keras"
MODEL_URL = "https://drive.google.com/uc?id=1vSIfD3viT5JSxpG4asA8APCwK0JK9Dvu"
CLASS_NAMES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# Baixar modelo, se necessário
if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    print("🔽 Baixando modelo...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Carregar o modelo
print("✅ Carregando modelo...")
model = load_model(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada."}), 400

    file = request.files["file"]
    try:
        image = Image.open(BytesIO(file.read())).convert("RGB")
        image = image.resize((256, 256))
        img_array = np.array(image)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "🧠 Eye Disease Classifier API"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
