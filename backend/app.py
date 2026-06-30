"""
app.py
Flask REST API that serves the trained brain tumor classification model
to the React frontend.

Endpoints:
    GET  /api/health         -> simple health check
    POST /api/predict        -> accepts an image file, returns prediction JSON

Run:
    python app.py
"""

import io
import os

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.models import load_model

from model.model_arch import CLASS_NAMES, IMG_SIZE

app = Flask(__name__)
CORS(app)  # allow requests from the React dev server

MODEL_PATH = os.environ.get("MODEL_PATH", "saved_model/brain_tumor_model.h5")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

model = None  # lazy-loaded on first request


def load_trained_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at '{MODEL_PATH}'. "
                f"Run train.py first to produce a model file."
            )
        model = load_model(MODEL_PATH)
    return model


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use form field 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use png/jpg/jpeg."}), 400

    try:
        net = load_trained_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    try:
        img_array = preprocess_image(file.read())
        preds = net.predict(img_array)[0]
        top_idx = int(np.argmax(preds))

        response = {
            "predicted_class": CLASS_NAMES[top_idx],
            "confidence": round(float(preds[top_idx]) * 100, 2),
            "probabilities": {
                CLASS_NAMES[i]: round(float(preds[i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            },
            "has_tumor": CLASS_NAMES[top_idx] != "no_tumor",
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
