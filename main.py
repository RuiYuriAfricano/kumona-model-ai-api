import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
import gdown
import os

# Caminho do modelo
MODEL_PATH = "best_model.keras"
MODEL_URL = "https://drive.google.com/uc?id=1vSIfD3viT5JSxpG4asA8APCwK0JK9Dvu"  # substitua pelo seu ID

# Baixar modelo se necessário
#if not os.path.exists(MODEL_PATH):
st.write("🔽 Baixando modelo...")
gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Carregar modelo
st.write("✅ Carregando modelo...")
model = load_model(MODEL_PATH)
class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# Interface
st.title("Eye Disease Classifier")
st.write("Faça upload de uma imagem do olho para detectar doenças.")

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagem carregada", use_column_width=True)

    image = image.resize((256, 256))
    img_array = np.array(image)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    st.subheader("Resultado:")
    st.write(f"**Doença detectada:** {predicted_class}")
    st.write(f"**Confiança:** {confidence:.2%}")
