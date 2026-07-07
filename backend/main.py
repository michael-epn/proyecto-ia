from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Cargar el modelo de mascotas
modelo = tf.keras.models.load_model("modelo_pets_custom.keras")
# 2. Las 37 clases del Oxford-IIIT Pet Dataset
CLASES_PETS = [
    'Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 
    'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair', 
    'Chihuahua', 'Egyptian Mau', 'English Cocker Spaniel', 'English Setter', 
    'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond', 
    'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 'Persian', 
    'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Samoyed', 'Scottish Terrier', 
    'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier', 'Wheaten Terrier', 
    'Yorkshire Terrier'
]

@app.post("/predecir/")
async def predecir(file: UploadFile = File(...)):
    image_data = await file.read()
    
    # 3. Preprocesamiento: RGB (color) y 128x128
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = image.resize((128, 128))
    
    # Estandarizar valores (0 a 1) y dar formato (1, 128, 128, 3)
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(-1, 128, 128, 3)

    # 4. Predicción
    prediccion = modelo.predict(img_array)
    categoria_idx = np.argmax(prediccion[0])
    
    # Obtener el porcentaje de confianza
    confianza = float(np.max(prediccion[0])) * 100

    # 5. Devolver JSON con nombre de raza y porcentaje
    return {
        "raza": CLASES_PETS[categoria_idx],
        "confianza": round(confianza, 2)
    }