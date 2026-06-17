from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import onnxruntime as ort
import io
import os
import urllib.request

app = Flask(__name__)

CLAVE_PROXY = os.environ.get('CLAVE_PROXY', 'cambiar_esto')

# MobileCLIP2 S0 - modelo ultra liviano (43MB), ideal para entornos con poca RAM
MODELO_URL = 'https://huggingface.co/plhery/mobileclip2-onnx/resolve/main/onnx/s0/vision_model.onnx'
MODELO_PATH = '/tmp/clip_vision.onnx'
TAMANO_IMAGEN = 256  # S0 usa 256x256

session = None

def cargar_modelo():
    global session
    if session is not None:
        return session

    if not os.path.exists(MODELO_PATH):
        print("Descargando modelo ONNX (MobileCLIP2 S0, ~43MB)...")
        urllib.request.urlretrieve(MODELO_URL, MODELO_PATH)
        print("Modelo descargado.")

    print("Cargando sesión ONNX Runtime...")
    opciones = ort.SessionOptions()
    opciones.intra_op_num_threads = 1
    opciones.inter_op_num_threads = 1
    session = ort.InferenceSession(MODELO_PATH, sess_options=opciones, providers=['CPUExecutionProvider'])
    print("Sesión ONNX cargada correctamente.")
    return session


def preprocesar_imagen(img):
    """Preprocesamiento para MobileCLIP2: resize a 256x256, mean=0 std=1 (solo escala 0-1)."""
    img = img.convert("RGB").resize((TAMANO_IMAGEN, TAMANO_IMAGEN), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0  # mean=(0,0,0), std=(1,1,1) → solo normalizar a [0,1]
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr


@app.route('/')
def home():
    return jsonify({'status': 'ok', 'mensaje': 'Proxy de embeddings activo (MobileCLIP2 ONNX)'})


@app.route('/embedding', methods=['POST'])
def generar_embedding():
    clave = request.form.get('clave', '')
    if clave != CLAVE_PROXY:
        return jsonify({'ok': False, 'error': 'Clave incorrecta'}), 403

    if 'foto' not in request.files:
        return jsonify({'ok': False, 'error': 'No se recibió ninguna foto'}), 400

    try:
        sess = cargar_modelo()

        foto = request.files['foto']
        img = Image.open(io.BytesIO(foto.read()))
        entrada = preprocesar_imagen(img)

        nombre_input = sess.get_inputs()[0].name
        resultado = sess.run(None, {nombre_input: entrada})
        embedding = resultado[0][0]

        # El modelo devuelve embeddings sin normalizar - normalizamos para similitud coseno
        norma = np.linalg.norm(embedding)
        if norma > 0:
            embedding = embedding / norma

        return jsonify({'ok': True, 'embedding': embedding.tolist()})

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Error procesando imagen: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
