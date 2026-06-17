from flask import Flask, request, jsonify
import torch
import open_clip
from PIL import Image
import io
import os
import gc

# Forzar uso de CPU únicamente (Render free tier no tiene GPU)
torch.set_num_threads(1)
device = torch.device('cpu')

app = Flask(__name__)

CLAVE_PROXY = os.environ.get('CLAVE_PROXY', 'cambiar_esto')

# Cargar el modelo CLIP una sola vez al iniciar el servicio
print("Cargando modelo CLIP (open_clip)...")
model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')
model.to(device)
model.eval()
gc.collect()
print("Modelo CLIP cargado correctamente.")


@app.route('/')
def home():
    return jsonify({'status': 'ok', 'mensaje': 'Proxy de embeddings activo (open_clip)'})


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
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        embedding = features[0].tolist()

        del img_tensor, features
        gc.collect()

        return jsonify({'ok': True, 'embedding': embedding})

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Error procesando imagen: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
