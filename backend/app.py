from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import os

app = Flask(__name__, 
            static_folder='../frontend/static',
            template_folder='../frontend/templates')
CORS(app)

model_path = os.path.join('model', 'saved_model', 'spam_model.pkl')
vectorizer_path = os.path.join('model', 'saved_model', 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        message = data['message']
        
        message_features = vectorizer.transform([message])
        prediction = model.predict(message_features)[0]
        probabilities = model.predict_proba(message_features)[0]
        
        if prediction == 1:
            result = "SPAM"
            confidence = float(probabilities[1])
        else:
            result = "NOT SPAM"
            confidence = float(probabilities[0])
        
        return jsonify({
            'message': message,
            'prediction': result,
            'confidence': round(confidence * 100, 2),
            'is_spam': bool(prediction == 1)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Server running on http://localhost:5000")
    app.run(debug=True, port=5000)