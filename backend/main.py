from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score

app = FastAPI(
    title="Asistente de Clasificación y Entrenamiento de Mascotas IA",
    description="Backend para clasificar razas de perros/gatos y reentrenar hiperparámetros en tiempo real."
)

# Habilitar CORS para conectar con el frontend sin problemas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Cargar el modelo pre-entrenado globalmente
MODEL_PATH = "modelo_pets_custom.keras"
try:
    modelo = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error al cargar el modelo base: {e}. Asegúrate de tener el archivo {MODEL_PATH} en el mismo directorio.")
    modelo = None

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

# Estructura para recibir los hiperparámetros desde el Frontend
class Hiperparametros(BaseModel):
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 0.001

@app.post("/predecir/")
async def predecir(file: UploadFile = File(...)):
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado en el servidor.")
        
    image_data = await file.read()
    
    # Preprocesamiento: RGB (color) y 128x128
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = image.resize((128, 128))
    
    # Estandarizar valores (0 a 1) y dar formato (1, 128, 128, 3)
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(-1, 128, 128, 3)

    # Predicción
    prediccion = modelo.predict(img_array)
    categoria_idx = np.argmax(prediccion[0])
    
    # Obtener el porcentaje de confianza
    confianza = float(np.max(prediccion[0])) * 100

    return {
        "raza": CLASES_PETS[categoria_idx],
        "confianza": round(confianza, 2)
    }

# ==========================================================
# ¡NUEVO ENDPOINT! CUMPLE CON LOS LITERALES B y C DE LA GUÍA
# ==========================================================
@app.post("/entrenar/")
async def entrenar(params: Hiperparametros):
    global modelo
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo base no encontrado para reentrenar.")

    try:
        # 1. Cargar una porción pequeña del dataset para un entrenamiento rápido en vivo
        # Cargamos solo el 5% para entrenamiento y el 5% para validación para no colgar la PC
        dataset = tfds.load('oxford_iiit_pet', split=['train[:5%]', 'test[:5%]'], as_supervised=True)
        train_ds_mini, test_ds_mini = dataset[0], dataset[1]

        IMG_SIZE = 128

        # Función de preprocesamiento local
        def preprocesar_local(imagen, etiqueta):
            imagen = tf.cast(imagen, tf.float32) / 255.0 
            imagen = tf.image.resize(imagen, (IMG_SIZE, IMG_SIZE))
            return imagen, etiqueta

        # Preparar los datasets optimizados
        train_ds_mini = train_ds_mini.map(preprocesar_local).shuffle(200).batch(params.batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds_mini = test_ds_mini.map(preprocesar_local).batch(params.batch_size).prefetch(tf.data.AUTOTUNE)

        # 2. Re-compilar el modelo usando el learning rate que mande el frontend
        optimizador = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
        modelo.compile(
            optimizer=optimizador, 
            loss='sparse_categorical_crossentropy', 
            metrics=["accuracy"]
        )

        # 3. Entrenar en vivo (pocas épocas, pocos datos = súper rápido)
        print(f"Iniciando reentrenamiento rápido: Epochs={params.epochs}, Batch Size={params.batch_size}, LR={params.learning_rate}")
        modelo.fit(train_ds_mini, epochs=params.epochs, verbose=0)

        # Guardar el modelo actualizado
        modelo.save(MODEL_PATH)

        # 4. Calcular dinámicamente las nuevas métricas tras el entrenamiento
        y_true, y_pred = [], []
        for imagenes, etiquetas in test_ds_mini:
            predicciones = modelo.predict(imagenes, verbose=0)
            y_pred.extend(np.argmax(predicciones, axis=1))
            y_true.extend(etiquetas.numpy())

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)

        # Devolver las métricas reales resultantes de la configuración del usuario
        return {
            "mensaje": "Modelo reentrenado con éxito en vivo",
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "parametros_usados": {
                "epochs": params.epochs,
                "batch_size": params.batch_size,
                "learning_rate": params.learning_rate
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante el entrenamiento: {str(e)}")