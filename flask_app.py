import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
import tensorflow as tf # Required for loading the model structure

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

# 2. تحميل النموذج
try:
    # 💡 نقوم بتحميل النموذج مرة واحدة
    MODEL = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    # هذا الخطأ هو سبب المشاكل السابقة، ويجب التعامل معه هنا
    print(f"FATAL ERROR: Could not load Keras model: {e}")
    # إذا فشل التحميل، لن نقوم بتشغيل التطبيق
    MODEL = None

# ----------------------------------------------
# 3. وظيفة معالجة الصورة
# ----------------------------------------------
def preprocess_image(image_file_bytes):
    if MODEL is None:
        return None
        
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
    if MODEL is None:
        return jsonify({'success': False, 'message': 'AI Model is not loaded on the server.'}), 500
        
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image file provided in the request'}), 400
    
    try:
        image_file = request.files['image'].read()
        processed_image = preprocess_image(image_file)
        
        # إجراء التنبؤ
        predictions = MODEL.predict(processed_image, verbose=0)[0]
        predicted_index = np.argmax(predictions)
        
        # 🔥🔥 الحل الحاسم: التحويل إلى float قياسي قبل الإرسال 🔥🔥
        confidence_score = float(np.max(predictions)) 

        # 💡 يجب أن يكون الرد JSON فقط (String, int, float)
        return jsonify({
            'success': True,
            'fault_class': CLASS_NAMES[predicted_index],
            'confidence': confidence_score
        })
        
    except Exception as e:
        # إذا حدث أي خطأ برمجي، نعرضه في السجل
        return jsonify({'success': False, 'message': f'Prediction processing failed: {str(e)}'}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "AI Server Operational (Waiting for POST on /predict_fault)"})
