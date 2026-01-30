from flask import Flask, render_template, request, redirect, flash, url_for
from flask_sqlalchemy import SQLAlchemy
import pickle
import numpy as np
import os
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize Database
db = SQLAlchemy(app)

# --- Database Model ---
class UserActivity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input_data = db.Column(db.String(500), nullable=False)
    prediction_result = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# --- Load Models ---
crop_model = None
yield_model = None
disease_model = None
class_indices = None

def load_models_at_start():
    global crop_model, yield_model, disease_model, class_indices
    try:
        # Load Crop Model
        if os.path.exists('models/crop_recommender.pkl'):
            with open('models/crop_recommender.pkl', 'rb') as f:
                crop_model = pickle.load(f)
            
        # Load Yield Model
        if os.path.exists('models/yield_predictor.pkl'):
            with open('models/yield_predictor.pkl', 'rb') as f:
                yield_model = pickle.load(f)
            
        # Load Disease Model
        if os.path.exists('models/disease_model.h5'):
            print("Loading Disease Model...")
            disease_model = load_model('models/disease_model.h5')
            print("Disease Model Loaded.")
            
        # Load Class Indices (The names of the diseases)
        if os.path.exists('models/class_indices.json'):
            with open('models/class_indices.json', 'r') as f:
                indices = json.load(f)
                # Swap keys and values so we can look up by number: {0: 'Apple_Rot'}
                class_indices = {v: k for k, v in indices.items()}
                print("Disease Class Names Loaded.")
                
    except Exception as e:
        print(f"Error loading models: {e}")

load_models_at_start()

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict-crop', methods=['GET', 'POST'])
def predict_crop():
    if request.method == 'POST':
        try:
            features = [
                float(request.form['N']),
                float(request.form['P']),
                float(request.form['K']),
                float(request.form['temperature']),
                float(request.form['humidity']),
                float(request.form['ph']),
                float(request.form['rainfall'])
            ]
            final_features = [np.array(features)]
            prediction = crop_model.predict(final_features)
            result = prediction[0]
            
            new_activity = UserActivity(input_data=str(features), prediction_result=f"Crop: {result}")
            db.session.add(new_activity)
            db.session.commit()
            
            return render_template('result.html', prediction=result, title="Recommended Crop")
        except Exception as e:
            flash(f"Error: {e}", "danger")
            return redirect(url_for('predict_crop'))
    return render_template('crop.html')

@app.route('/predict-yield', methods=['GET', 'POST'])
def predict_yield():
    if request.method == 'POST':
        try:
            features = [
                float(request.form['Area']),
                float(request.form['Rainfall']),
                float(request.form['Pest_Count'])
            ]
            final_features = [np.array(features)]
            prediction = yield_model.predict(final_features)
            result = round(prediction[0], 2)
            
            new_activity = UserActivity(input_data=str(features), prediction_result=f"Yield: {result} tons")
            db.session.add(new_activity)
            db.session.commit()
            
            return render_template('result.html', prediction=f"{result} Tons", title="Predicted Yield")
        except Exception as e:
            flash(f"Error: {e}", "danger")
            return redirect(url_for('predict_yield'))
    return render_template('yield.html')

@app.route('/detect-disease', methods=['GET', 'POST'])
def detect_disease():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file:
            try:
                # Save the uploaded file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Check if model is loaded
                if disease_model is None or class_indices is None:
                    result_text = "Error: Model not loaded. Did you run train_models.py?"
                else:
                    # Preprocess the image to match the training size (128x128)
                    img = image.load_img(filepath, target_size=(128, 128))
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array /= 255.0  # Normalize pixel values
                    
                    # Make Prediction
                    predictions = disease_model.predict(img_array)
                    predicted_index = np.argmax(predictions[0])
                    
                    # Get the Disease Name from the index
                    disease_name = class_indices.get(predicted_index, "Unknown")
                    confidence = round(100 * np.max(predictions[0]), 2)
                    
                    result_text = f"{disease_name} (Confidence: {confidence}%)"

                # Save to Database
                new_activity = UserActivity(input_data=f"Image: {filename}", prediction_result=result_text)
                db.session.add(new_activity)
                db.session.commit()
                
                return render_template('result.html', prediction=result_text, title="Disease Detection")
                
            except Exception as e:
                print(f"Prediction Error: {e}")
                flash(f"Error processing image: {e}", "danger")
                return redirect(request.url)
            
    return render_template('disease.html')

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
