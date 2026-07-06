const input = document.getElementById("imagenInput");
const preview = document.getElementById("preview");
const textoPreview = document.getElementById("textoPreview");
const resultado = document.getElementById("resultado");

const API_URL = "http://127.0.0.1:8000/predecir/";

// 1. Mostrar vista previa y estado de preparación
input.addEventListener("change", () => {
  const archivo = input.files[0];

  if (archivo) {
    // createObjectURL crea una URL temporal en memoria para la imagen
    preview.src = URL.createObjectURL(archivo);
    preview.classList.remove("oculto");
    textoPreview.style.display = "none";

    resultado.innerHTML = `
      <div class="estrella">🐾</div>
      <div>
        <strong>Imagen cargada</strong>
        <p>Lista para analizar</p>
        <span>Presiona el botón para predecir</span>
      </div>
    `;
  }
});

// 2. Consumir la API
async function enviarImagen() {
  if (input.files.length === 0) {
    alert("Por favor, selecciona una imagen primero.");
    return;
  }

  const formData = new FormData();
  formData.append("file", input.files[0]);

  // Mostrar estado de carga mientras el modelo procesa
  resultado.innerHTML = `
    <div class="estrella">⏳</div>
    <div>
      <strong>Analizando...</strong>
      <p>La IA está revisando la imagen</p>
      <span>Espera un momento</span>
    </div>
  `;

  try {
    const respuesta = await fetch(API_URL, {
      method: "POST",
      body: formData
    });

    if (!respuesta.ok) throw new Error("Error en el servidor HTTP");

    const datos = await respuesta.json();

    // Renderizar el resultado exitoso (usando la clave 'raza' del backend)
    resultado.innerHTML = `
      <div class="estrella">⭐</div>
      <div>
        <strong>${datos.raza}</strong>
        <p>Raza identificada</p>
        <span>Confianza: ${datos.confianza}%</span>
      </div>
    `;

  } catch (error) {
    console.error("Error capturado:", error);

    resultado.innerHTML = `
      <div class="estrella">❌</div>
      <div>
        <strong>Error de conexión</strong>
        <p>No se pudo conectar con la API</p>
        <span>Verifica que FastAPI (Uvicorn) esté ejecutándose</span>
      </div>
    `;
  }
}