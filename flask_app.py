from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)


MODEL_PATH = "FINAL_API_MODEL.h5"


model = tf.keras.models.load_model(MODEL_PATH)


CLASSES = ["Corroded battery Terminals", "Healthy Battery", "Healthy Engine", "Healthy Tire","Low tire pressure","Oil Leak"]

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((416, 416))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image sent"}), 400

    image_file = request.files["image"]
    img_bytes = image_file.read()

    img = preprocess_image(img_bytes)
    predictions = model.predict(img)[0]
    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index]
    predicted_class = CLASSES[predicted_index]

    if confidence < 0.6:
        return jsonify({
            "result": "No visible fault detected. Please retake the image.",
            "confidence": float(confidence)
        })

    return jsonify({
        "result": f" Detected fault in {predicted_class}",
        "confidence": float(confidence)
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({"API is running successfully"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
