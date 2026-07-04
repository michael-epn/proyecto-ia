from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io

app = FastAPI()

# Esto permite que tu archivo HTML (frontend) pueda comunicarse con esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Cargar el modelo guardado al arrancar el servidor
modelo = tf.keras.models.load_model("modelo_mnist.h5")

@app.post("/predecir/")
async def predecir(file: UploadFile = File(...)):
    # 2. Leer la imagen que nos envía el HTML
    image_data = await file.read()
    
    # 3. Preparar la imagen para el modelo
    # Convertir a escala de grises
    image = Image.open(io.BytesIO(image_data)).convert('L')
    
    # MNIST fue entrenado con números blancos sobre fondo negro. 
    # Si subes un número negro en fondo blanco, necesitamos invertir los colores:
    image = ImageOps.invert(image) 
    
    # Redimensionar a 28x28 píxeles
    image = image.resize((28, 28))
    
    # Convertir a matriz matemática y estandarizar valores entre 0 y 1
    img_array = np.array(image).astype("float32") / 255.0
    
    # Darle la forma que espera la capa convolucional: (1, 28, 28, 1)
    img_array = img_array.reshape(-1, 28, 28, 1)

    # 4. Hacer la predicción
    prediccion = modelo.predict(img_array)
    categoria = np.argmax(prediccion[0])

    # 5. Devolver el resultado en formato JSON
    return {"numero_predicho": int(categoria)}