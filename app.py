from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# El token de Hugging Face se configura como variable de entorno en Render (más seguro)
HF_TOKEN = os.environ.get('HF_TOKEN', '')
HF_MODELO = 'https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32'

# Clave simple para que solo tu sitio pueda usar este proxy
CLAVE_PROXY = os.environ.get('CLAVE_PROXY', 'cambiar_esto')


@app.route('/')
def home():
    return jsonify({'status': 'ok', 'mensaje': 'Proxy de embeddings activo'})


@app.route('/embedding', methods=['POST'])
def generar_embedding():
    # Verificar clave
    clave = request.form.get('clave', '')
    if clave != CLAVE_PROXY:
        return jsonify({'ok': False, 'error': 'Clave incorrecta'}), 403

    if 'foto' not in request.files:
        return jsonify({'ok': False, 'error': 'No se recibió ninguna foto'}), 400

    foto = request.files['foto']
    imagen_bytes = foto.read()

    try:
        resp = requests.post(
            HF_MODELO,
            headers={
                'Authorization': f'Bearer {HF_TOKEN}',
                'Content-Type': 'application/octet-stream',
            },
            data=imagen_bytes,
            timeout=30,
        )

        if resp.status_code != 200:
            return jsonify({
                'ok': False,
                'error': f'HF respondió {resp.status_code}: {resp.text[:300]}'
            }), 502

        data = resp.json()

        # Normalizar el formato de respuesta (a veces viene anidado)
        embedding = data
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            embedding = data[0][0] if isinstance(data[0][0], list) else data[0]

        return jsonify({'ok': True, 'embedding': embedding})

    except requests.exceptions.RequestException as e:
        return jsonify({'ok': False, 'error': f'Error de conexión: {str(e)}'}), 502


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
