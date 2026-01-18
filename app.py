from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Model load karne ka sahi tareeka
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'medical_model.pkl')
model = joblib.load(model_path)
# Wahi symptoms jo train_model.py mein the
symptoms_list = ['Fever', 'Cough', 'Headache', 'cold' 'Fatigue']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_symptoms = data.get('symptoms', [])
    
    # User input ko numerical vector mein badlein (0 aur 1)
    input_vector = [1 if s in user_symptoms else 0 for s in symptoms_list]
    
    # AI Prediction
    prediction = model.predict([input_vector])[0]
    
    # Confidence (Probability) calculate karein
    proba = model.predict_proba([input_vector])
    confidence = f"{np.max(proba) * 100:.1f}%"

    return jsonify({
        "prediction": f"AI Result: {prediction}",
        "confidence": confidence,
        "analysis": "ML model analysis complete."
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)