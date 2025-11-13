from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(_name_)


MODEL_PATH = "FINAL_API_MODEL.h5"


model = tf.keras.models.load_model(MODEL_PATH)


CLASSES = ["Engine", "Brakes", "Suspension", "Transmission"]

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
            "result": "Re-capture image with better focus.",
            "confidence": float(confidence)
        })

    return jsonify({
        "result": f" Diagnosis successfully confirmed in {predicted_class}",
        "confidence": float(confidence)
    })

@app.route("/", methods=["GET"])
def home():
    return "API is running successfully"

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000)
