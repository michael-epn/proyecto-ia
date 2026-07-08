# 🐾 Clasificador de Mascotas con IA

Este repositorio contiene una aplicación Full-Stack que utiliza Inteligencia Artificial para clasificar imágenes de mascotas (perros y gatos) en 37 razas diferentes. Es el proyecto del Segundo Bimestre para la materia de **Fundamentos de la Inteligencia Artificial**.

## 🚀 Características Principales

* **Interfaz Intuitiva:** Frontend responsivo y amigable para subir o arrastrar imágenes de mascotas de forma sencilla.
* **Predicción en Tiempo Real:** Comunicación ágil con una API RESTful para devolver la raza identificada y el porcentaje de confianza.
* **Modelo Pre-entrenado:** Uso de *Transfer Learning* con la arquitectura **MobileNetV2** para optimizar el rendimiento y la precisión de la clasificación.
* **Métricas del Modelo:**
    * **Accuracy:** 84.06%
    * **Precisión:** 84.41%
    * **Recall:** 84.01%

## 🛠️ Tecnologías y Librerías Utilizadas

**Frontend:**
* HTML5, CSS3, JavaScript (Vanilla)
* Consumo de API con `fetch` nativo.

**Backend & Machine Learning:**
* **Python:** Lenguaje base.
* **FastAPI / Uvicorn:** Creación del servidor web y endpoints de alta velocidad.
* **TensorFlow & Keras:** Entrenamiento, guardado (`.keras`) y carga del modelo de red neuronal.
* **TensorFlow Datasets (TFDS):** Carga del dataset *Oxford-IIIT Pet Dataset* (~7,000 imágenes).
* **Scikit-Learn:** Cálculo de métricas finales (Accuracy, Precision, Recall).
* **Pillow (PIL):** Preprocesamiento de imágenes recibidas antes de la predicción.

## ⚙️ Instalación y Ejecución Local

Sigue estos pasos para levantar el proyecto en tu entorno local:

### 1. Clonar el repositorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd proyecto-ia
