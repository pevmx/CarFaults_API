import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
import sys
# 🔥 سنستخدم مكتبة TensorFlow Lite (لتوفير الذاكرة)
import tensorflow.lite as tflite 
from tensorflow.lite.python.interpreter import Interpreter # لاستخدام مفسر TFLite

app = Flask(__name__)

# ----------------------------------------------
# 1. SETUP AND CONSTANTS
# ----------------------------------------------
MODEL_PATH = 'API_DEPLOYMENT_TFLITE_FINAL.tflite' 
IMAGE_SIZE = (416, 416)
CLASS_NAMES = [
    'Corroded battery Terminals', 'Oil Leak', 'Low tire pressure', 
    'Healthy Battery', 'Healthy Engine', 'Healthy Tire'
] 

# 2. تحميل المفسر (Interpreter)
try:
    # تحميل المفسر (Interpreter) الذي يستهلك ذاكرة أقل
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLite Model Interpreter loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load TFLite model: {e}")
    sys.exit(1)


# ----------------------------------------------
# 3. وظيفة معالجة الصورة
# ----------------------------------------------
def preprocess_image(image_file_bytes):
    """Loads image bytes, resizes, and prepares it for TFLite model."""
    image = Image.open(io.BytesIO(image_file_bytes)).convert('RGB')
    image = image.resize(IMAGE_SIZE)
    input_data = np.asarray(image, dtype=np.float32)
    input_data = input_data / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

# ----------------------------------------------
# 4. نقطة النهاية للذكاء الاصطناعي (API Endpoint)
# ----------------------------------------------
@app.route('/predict_fault', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image file provided in the request'}), 400
    
    try:
        image_file = request.files['image'].read()
        processed_image = preprocess_image(image_file)
        
        # 💡 تشغيل المفسر (Running the TFLite Interpreter)
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        
        # الحصول على النتيجة
        output_tensor = interpreter.get_tensor(output_details[0]['index'])
        predictions = output_tensor[0]
        
        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence_score = float(np.max(predictions)) # تحويل إلى float قياسي

        return jsonify({
            'success': True,
            'fault_class': predicted_class, 
            'confidence': confidence_score
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Prediction processing failed: {str(e)}'}), 500

# ----------------------------------------------
# 5. تشغيل التطبيق
# ----------------------------------------------
# (يجب أن يتم تشغيل هذا عبر Gunicorn على Render)
