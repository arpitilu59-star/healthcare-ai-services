from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get('symptoms', [])
    
    # Simple AI Logic (Baad mein isme ML Model .pkl file add karenge)
    if "Fever" in symptoms and "Cough" in symptoms:
        prediction = "Potential Viral Infection"
        confidence = "85%"
    elif "Headache" in symptoms:
        prediction = "Tension Headache / Migraine"
        confidence = "70%"
    else:
        prediction = "Symptoms Analyzed: Please consult a physician."
        confidence = "60%"

    return jsonify({
        "prediction": prediction,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(port=5000)