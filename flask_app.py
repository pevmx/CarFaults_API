import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import sys

app = Flask(__name__)

# ----------------------------------------------
# 1. SETUP AND CONSTANTS
# ----------------------------------------------
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

# تحميل النموذج (يتم مرة واحدة عند بدء تشغيل السيرفر)
try:
    # 🔥 نعتمد على تحميل TensorFlow الكامل هنا
    MODEL = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    # إيقاف السيرفر إذا فشل تحميل النموذج
    sys.exit(1)

# ----------------------------------------------
# 2. وظيفة معالجة الصورة
# ----------------------------------------------
def preprocess_image(image_file_bytes):
    """Loads image bytes, resizes, and prepares it for the model."""
    if MODEL is None:
        return None
        
    image = Image.open(io.BytesIO(image_file_bytes)).convert('RGB')
    image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32)
    image_array = image_array / 255.0
    return np.expand_dims(image_array, axis=0)

# ----------------------------------------------
# 3. نقطة النهاية للذكاء الاصطناعي (API Endpoint)
# ----------------------------------------------
@app.route('/predict_fault', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image file provided in the request'}), 400
    
    try:
        image_file = request.files['image'].read()
        processed_image = preprocess_image(image_file)
        
        predictions = MODEL.predict(processed_image, verbose=0)[0]
        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence_score = float(np.max(predictions))
        
        # 💡 فحص الثقة للتأكد من جودة الصورة 
        if confidence_score < 0.6:
            return jsonify({
                "success": False,
                "message": "Confidence threshold not met. Please recapture the image.",
                "confidence": confidence_score
            }), 400

        return jsonify({
            'success': True,
            'fault_class': predicted_class, 
            'confidence': confidence_score
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Prediction processing failed: {str(e)}'}), 500
