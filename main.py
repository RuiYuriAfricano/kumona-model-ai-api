import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import load_model
import gdown
import os

# Caminho do modelo
MODEL_PATH = "best_model.keras"
MODEL_URL = "https://drive.google.com/uc?id=1vSIfD3viT5JSxpG4asA8APCwK0JK9Dvu"  # substitua pelo seu ID

# Baixar modelo se necessário
if not os.path.exists(MODEL_PATH):
    st.write("🔽 Baixando modelo...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Carregar modelo
st.write("✅ Carregando modelo...")
model = load_model(MODEL_PATH)
class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# Função de predição
def predict(model, img):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        img.save(tmp_file, format="JPG")
        temp_path = tmp_file.name

    img2 = tf.keras.utils.load_img(temp_path, target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(img2)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)  # Yüzde olarak, 2 ondalık
    return predicted_class, confidence

# Interface
st.title("Eye Disease Classifier")
st.write("Faça upload de uma imagem do olho para detectar doenças.")

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem carregada", use_container_width=True)

    # Fazer a predição
    predicted_class, confidence = predict(model, image)
    
    st.subheader("Resultado:")
    st.write(f"**Doença detectada:** {predicted_class}")
    st.write(f"**Confiança:** {confidence:.2f}%")
