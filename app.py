import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# ── Configuración de la página ──────────────────────────────────────────────
st.set_page_config(page_title="Clasificador de Flores", page_icon="🌸", layout="centered")

# ── Cargar modelo ────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelo_flores.keras")

model = cargar_modelo()
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
nombres_es  = ['Margarita', 'Diente de León', 'Rosa', 'Girasol', 'Tulipán']

# ── Función de predicción ────────────────────────────────────────────────────
def predecir(imagen_pil):
    """Recibe una imagen PIL, la redimensiona a 128x128, normaliza y predice."""
    img = imagen_pil.convert("RGB").resize((128, 128))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    return preds

def mostrar_resultados(preds):
    """Muestra la distribución de probabilidades con la predicción resaltada."""
    idx_max = int(np.argmax(preds))
    st.subheader("📊 Distribución de probabilidades")
    for i, (eng, esp, prob) in enumerate(zip(class_names, nombres_es, preds)):
        label = f"{esp} ({eng}): {prob*100:.1f}%"
        if i == idx_max:
            st.success(f"⭐ PREDICCIÓN: {label}")
        else:
            st.progress(float(prob), text=label)

# ── Título ───────────────────────────────────────────────────────────────────
st.title("🌸 Clasificador de Flores")
st.write("Sistema de visión artificial para catalogación de flora — CNN desde cero")

# ── Catálogo de clases ───────────────────────────────────────────────────────
st.subheader("📚 Catálogo de clases")
cols = st.columns(5)
iconos = ["🌼", "🌻", "🌹", "🌞", "🌷"]
for i, (eng, esp, ico) in enumerate(zip(class_names, nombres_es, iconos)):
    cols[i].info(f"{ico}\n\n**{esp}**\n\n_{eng}_")

st.divider()

# ── Selector de fuente de imagen ─────────────────────────────────────────────
st.subheader("📷 Fuente de la imagen")
opcion = st.radio(
    "¿Cómo quieres ingresar la imagen?",
    ["📁 Subir archivo", "🌐 URL de internet", "📸 Cámara"],
    horizontal=True
)

imagen_pil = None

# ── OPCIÓN 1: Subir archivo ──────────────────────────────────────────────────
if opcion == "📁 Subir archivo":
    uploaded = st.file_uploader(
        "Sube una imagen de flor",
        type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded:
        imagen_pil = Image.open(uploaded)

# ── OPCIÓN 2: URL de internet ────────────────────────────────────────────────
elif opcion == "🌐 URL de internet":
    url = st.text_input("Pega la URL de una imagen de flor:", placeholder="https://ejemplo.com/flor.jpg")
    if url:
        try:
            with st.spinner("Descargando imagen..."):
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                imagen_pil = Image.open(BytesIO(response.content))
            st.success("Imagen descargada correctamente")
        except requests.exceptions.RequestException as e:
            st.error(f"No se pudo descargar la imagen. Verifica la URL.\nError: {e}")
        except Exception as e:
            st.error(f"Error al procesar la imagen: {e}")

# ── OPCIÓN 3: Cámara ─────────────────────────────────────────────────────────
elif opcion == "📸 Cámara":
    foto = st.camera_input("Toma una foto de una flor")
    if foto:
        imagen_pil = Image.open(foto)

# ── Mostrar imagen y predicción ───────────────────────────────────────────────
if imagen_pil is not None:
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Imagen original")
        st.image(imagen_pil, use_column_width=True)
        st.caption(f"Tamaño original: {imagen_pil.size[0]}x{imagen_pil.size[1]} px")

    with col2:
        st.subheader("🔍 Imagen procesada (128x128)")
        img_resized = imagen_pil.convert("RGB").resize((128, 128))
        st.image(img_resized, use_column_width=True)
        st.caption("Redimensionada a 128x128 para el modelo")

    st.divider()

    with st.spinner("Analizando imagen..."):
        preds = predecir(imagen_pil)

    mostrar_resultados(preds)

    st.divider()
    st.caption("Modelo CNN entrenado desde cero con el Flower Photos Dataset de TensorFlow · 5 clases · ~73% accuracy en validación")