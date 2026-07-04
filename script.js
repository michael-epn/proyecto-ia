const input = document.getElementById("imagenInput");
const preview = document.getElementById("preview");
const textoPreview = document.getElementById("textoPreview");
const resultado = document.getElementById("resultado");

input.addEventListener("change", () => {
  const archivo = input.files[0];

  if (archivo) {
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

async function enviarImagen() {
  if (input.files.length === 0) {
    alert("Por favor, selecciona una imagen primero.");
    return;
  }

  const formData = new FormData();
  formData.append("file", input.files[0]);

  resultado.innerHTML = `
    <div class="estrella">?</div>
    <div>
      <strong>Analizando...</strong>
      <p>La IA está revisando la imagen</p>
      <span>Espera un momento</span>
    </div>
  `;

  try {
    const respuesta = await fetch("http://127.0.0.1:8000/predecir/", {
      method: "POST",
      body: formData
    });

    const datos = await respuesta.json();

    resultado.innerHTML = `
      <div>
        <strong>${datos.clase_predicha}</strong>
        <p>Raza o categoría predicha</p>
        <span>Confianza: ${datos.confianza}%</span>
      </div>
    `;

  } catch (error) {
    console.error(error);

    resultado.innerHTML = `
      <div>
        <strong>Error</strong>
        <p>No se pudo conectar con la API</p>
        <span>Activa el backend con FastAPI</span>
      </div>
    `;
  }
}