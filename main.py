# main.py - Banana Disease Detection API
from flask import Flask, request, jsonify
import joblib
import numpy as np
from PIL import Image
import io

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# -----------------------------
# LOAD MODEL SAAT SERVER START
# -----------------------------
print("⏳ Memuat model CNN-SVM...")

try:
    # Muat pipeline dari file .pkl
    bundle = joblib.load("model/cnn_svm_full.pkl")
    scaler = bundle["scaler"]
    pca = bundle["pca"]
    svm_model = bundle["model"]
    classes = bundle["classes"]
    print(f"✅ Kelas: {classes}")
except Exception as e:
    print(f"❌ Gagal muat model: {e}")
    raise e

# Load CNN: MobileNetV2 (hanya sekali)
print("⏳ Memuat MobileNetV2...")
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

cnn = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
print("✅ Server siap!")


@app.route('/')
def home():
    return """
    <h1>🍌 API Deteksi Penyakit Daun Pisang</h1>
    <p>Kirim gambar ke <code>/predict</code> via POST (form-data, field: 'file')</p>
    """


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded. Gunakan field "file".'}), 400

    file = request.files['file']
    
    try:
        # Baca gambar
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((224, 224))
        arr = img_to_array(img)
        arr = preprocess_input(arr)  # [-1, 1]
        
        # Ekstraksi fitur
        feat = cnn.predict(np.expand_dims(arr, axis=0), verbose=0)[0].reshape(1, -1)
        
        # Pipeline: Scaler → PCA → SVM
        feat_scaled = scaler.transform(feat)
        if pca is not None:
            feat_final = pca.transform(feat_scaled)
        else:
            feat_final = feat_scaled
        
        # Prediksi
        pred_idx = svm_model.predict(feat_final)[0]
        proba = svm_model.predict_proba(feat_final)[0]

        result = {
            'class': classes[int(pred_idx)],
            'confidence': float(np.max(proba)),
            'probabilities': {cls: float(p) for cls, p in zip(classes, proba)},
            'success': True
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Jalankan server (untuk lokal)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)