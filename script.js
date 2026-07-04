const input = document.getElementById("imagenInput");
const preview = document.getElementById("preview");
const resultado = document.getElementById("resultado");

input.addEventListener("change", () => {
  const archivo = input.files[0];

  if (archivo) {
    preview.src = URL.createObjectURL(archivo);
    preview.classList.remove("oculto");
    resultado.innerText = "Imagen cargada. Presiona predecir.";
  }
});

async function enviarImagen() {
  if (input.files.length === 0) {
    alert("Por favor, selecciona una imagen primero.");
    return;
  }

  const formData = new FormData();
  formData.append("file", input.files[0]);

  resultado.innerText = "Analizando imagen...";

  try {
    const respuesta = await fetch("http://127.0.0.1:8000/predecir/", {
      method: "POST",
      body: formData
    });

    const datos = await respuesta.json();

    resultado.innerHTML = `
      Mascota predicha: <br>
      🐾 ${datos.clase_predicha} <br>
      Confianza: ${datos.confianza}%
    `;

  } catch (error) {
    console.error(error);
    resultado.innerText = "No se pudo conectar con la API.";
  }
}