from flask import Flask, request, jsonify
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import os

app = Flask(__name__)

CLAVE_PROXY = os.environ.get('CLAVE_PROXY', 'cambiar_esto')

# Cargar el modelo CLIP una sola vez al iniciar el servicio (no por cada request)
print("Cargando modelo CLIP...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print("Modelo CLIP cargado correctamente.")


@app.route('/')
def home():
    return jsonify({'status': 'ok', 'mensaje': 'Proxy de embeddings activo (CLIP local)'})


@app.route('/embedding', methods=['POST'])
def generar_embedding():
    clave = request.form.get('clave', '')
    if clave != CLAVE_PROXY:
        return jsonify({'ok': False, 'error': 'Clave incorrecta'}), 403

    if 'foto' not in request.files:
        return jsonify({'ok': False, 'error': 'No se recibió ninguna foto'}), 400

    foto = request.files['foto']

    try:
        img = Image.open(io.BytesIO(foto.read())).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            features = model.get_image_features(**inputs)
            # Normalizar el vector (importante para la similitud coseno)
            features = features / features.norm(dim=-1, keepdim=True)

        embedding = features[0].tolist()
        return jsonify({'ok': True, 'embedding': embedding})

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Error procesando imagen: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
