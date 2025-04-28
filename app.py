from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('mnist_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and open image
        file = request.files['file']
        img = Image.open(io.BytesIO(file.read()))
        
        # Preprocess exactly like training data
        img = img.convert('L').resize((28, 28))
        img_array = np.array(img)
        
        # Invert and normalize
        img_array = 1.0 - (img_array / 255.0)  # MNIST-style (white digit on black)
        img_array = img_array.reshape(1, 28, 28, 1).astype('float32')
        
        # Predict
        predictions = model.predict(img_array)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        return jsonify({
            'digit': predicted_digit,
            'confidence': round(confidence * 100, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
