import streamlit as st
import numpy as np
from PIL import Image
import keras

CLASES = [
    {"nombre": "Tierra de Cultivo Anual",      "emoji": "🌾", "estado": "✅ Terreno natural"},
    {"nombre": "Tierra de Cultivo Permanente", "emoji": "🍇", "estado": "✅ Terreno natural"},
    {"nombre": "Bosque",                        "emoji": "🌲", "estado": "✅ Terreno natural"},
    {"nombre": "Autopista",                     "emoji": "🛣️", "estado": "ℹ️ Infraestructura"},
    {"nombre": "Tierra Herbácea",              "emoji": "🌿", "estado": "✅ Terreno natural"},
    {"nombre": "Zona Industrial",              "emoji": "🏭", "estado": "⚠️ Zona industrial"},
    {"nombre": "Masa de Agua",                 "emoji": "💧", "estado": "✅ Terreno natural"},
    {"nombre": "Zona Residencial",             "emoji": "🏘️", "estado": "ℹ️ Infraestructura"},
    {"nombre": "Río o Mar",                    "emoji": "🌊", "estado": "✅ Terreno natural"},
    {"nombre": "Vegetación Arbustiva",         "emoji": "🌵", "estado": "✅ Terreno natural"},
]

st.set_page_config(page_title="EuroSAT Clasificador", page_icon="🛰️", layout="centered")
st.title("🛰️ EuroSAT Clasificador")
st.caption("Análisis de imágenes satelitales con Inteligencia Artificial")
st.divider()

@st.cache_resource
def cargar_modelo():
    return keras.models.load_model("modelo_final_eurosat.keras")

with st.spinner("Cargando modelo de IA..."):
    modelo = cargar_modelo()

imagen_subida = st.file_uploader("📤 Sube una imagen satelital", type=["jpg", "jpeg", "png"])

if imagen_subida:
    imagen = Image.open(imagen_subida).convert("RGB")
    st.image(imagen, caption="Imagen subida", use_container_width=True)
    if st.button("🔍 Analizar imagen", use_container_width=True, type="primary"):
        with st.spinner("Analizando..."):
            img_resized = imagen.resize((224, 224))
            arr = np.array(img_resized, dtype=np.float32) / 255.0
            entrada = np.expand_dims(arr, axis=0)
            prediccion = modelo.predict(entrada, verbose=0)[0]
        idx = int(np.argmax(prediccion))
        confianza = float(prediccion[idx]) * 100
        clase = CLASES[idx]
        st.divider()
        st.subheader(f"{clase['emoji']} {clase['nombre']}")
        st.write(f"**Estado:** {clase['estado']}")
        st.progress(confianza / 100, text=f"Confianza: {confianza:.1f}%")
        st.divider()
        st.write("**📊 Todas las categorías:**")
        for i, prob in sorted(enumerate(prediccion), key=lambda x: x[1], reverse=True):
            c = CLASES[i]
            st.write(f"{c['emoji']} {c['nombre']}")
            st.progress(float(prob), text=f"{float(prob)*100:.1f}%")
