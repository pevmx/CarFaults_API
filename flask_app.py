import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
import tensorflow as tf

app = Flask(__name__)


MODEL_PATH = 'FINAL_API_MODEL.h5' 
IMAGE_SIZE = (416, 416)
CLASS_NAMES = [
    'Corroded battery Terminals', 
    'Oil Leak', 
    'Low tire pressure', 
    'Healthy Battery', 
    'Healthy Engine', 
    'Healthy Tire'
] 


try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Keras/TensorFlow Model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load Keras model: {e}")

def preprocess_image(image_file_bytes):
    """Loads image bytes, resizes, and prepares it for the model."""
    image = Image.open(io.BytesIO(image_file_bytes)).convert('RGB')
    image = image.resize(IMAGE_SIZE)
    
    input_data = np.asarray(image, dtype=np.float32)
    input_data = input_data / 255.0
    
    input_data = np.expand_dims(input_data, axis=0)
    return input_data


@app.route('/predict_fault', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image file provided in the request'}), 400
    
    try:
        image_file = request.files['image'].read()
        processed_image = preprocess_image(image_file)
        
        predictions = model.predict(processed_image, verbose=0)[0]
        
        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence_score = float(np.max(predictions))

        return jsonify({
            'success': True,
            'fault_class': predicted_class, 
            'confidence': confidence_score
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Prediction processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)