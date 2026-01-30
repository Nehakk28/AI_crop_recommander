import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

def load_or_generate_crop_data(n_samples=2000):
    """Loads data from 'Crop_recommendation.csv' or generates synthetic data."""
    csv_path = 'Crop_recommendation.csv'
    if os.path.exists(csv_path):
        print(f"Loading dataset from {csv_path}...")
        try:
            df = pd.read_csv(csv_path)
            print("Dataset loaded successfully.")
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}. Falling back to synthetic data.")

    print("CSV not found. Generating synthetic data...")
    np.random.seed(42)
    crops = ['Rice', 'Maize', 'Chickpea', 'Kidneybeans', 'Pigeonpeas']
    data = []
    
    ranges = {
        'Rice': {'N': (60, 90), 'P': (35, 60), 'K': (35, 45), 'temp': (20, 27), 'hum': (80, 90), 'ph': (5.5, 7.2), 'rain': (200, 300)},
        'Maize': {'N': (60, 100), 'P': (35, 60), 'K': (15, 25), 'temp': (18, 27), 'hum': (50, 70), 'ph': (5.5, 7.0), 'rain': (60, 100)},
        'Chickpea': {'N': (20, 60), 'P': (55, 80), 'K': (75, 85), 'temp': (17, 22), 'hum': (15, 20), 'ph': (6.0, 8.0), 'rain': (65, 85)},
        'Kidneybeans': {'N': (10, 40), 'P': (55, 80), 'K': (15, 25), 'temp': (15, 25), 'hum': (20, 30), 'ph': (5.5, 6.0), 'rain': (60, 150)},
        'Pigeonpeas': {'N': (10, 40), 'P': (55, 80), 'K': (15, 25), 'temp': (18, 29), 'hum': (30, 70), 'ph': (4.5, 7.5), 'rain': (90, 180)}
    }

    for _ in range(n_samples):
        crop = np.random.choice(crops)
        r = ranges[crop]
        row = [
            np.random.uniform(*r['N']),
            np.random.uniform(*r['P']),
            np.random.uniform(*r['K']),
            np.random.uniform(*r['temp']),
            np.random.uniform(*r['hum']),
            np.random.uniform(*r['ph']),
            np.random.uniform(*r['rain']),
            crop
        ]
        data.append(row)
    
    columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
    return pd.DataFrame(data, columns=columns)

def train_crop_recommender():
    print("Training Crop Recommender...")
    df = load_or_generate_crop_data(2000)
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Crop Recommender Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        
    with open('models/crop_recommender.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Saved models/crop_recommender.pkl")

def train_yield_predictor():
    print("\nTraining Yield Predictor...")
    np.random.seed(42)
    data = []
    for _ in range(500):
        area = np.random.uniform(1, 100)
        rainfall = np.random.uniform(500, 2000)
        pest_count = np.random.uniform(0, 50)
        yield_val = (area * 3.5) + (rainfall * 0.01) - (pest_count * 0.5) + np.random.normal(0, 2)
        data.append([area, rainfall, pest_count, yield_val])
    
    df = pd.DataFrame(data, columns=['Area', 'Rainfall', 'Pest_Count', 'Yield'])
    X = df.drop('Yield', axis=1)
    y = df['Yield']
    
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    with open('models/yield_predictor.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Saved models/yield_predictor.pkl")

def train_disease_model(dataset_path='train'):
    print(f"\nTraining Disease Detection Model using data from '{dataset_path}'...")
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset folder '{dataset_path}' not found!")
        print("Please make sure you have a folder named 'train' with your disease images.")
        return

    # Image parameters
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32

    # Data Augmentation (Normalizes and adds variety)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    # Load Training Data
    print("Loading training images...")
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # Load Validation Data
    print("Loading validation images...")
    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Save the class names (so app.py knows what 0, 1, 2 mean)
    with open('models/class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)
    print("Saved models/class_indices.json")

    num_classes = len(train_generator.class_indices)
    
    # Build CNN Model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    model.fit(
        train_generator,
        epochs=10, 
        validation_data=validation_generator
    )
    
    model.save('models/disease_model.h5')
    print("Saved models/disease_model.h5")

if __name__ == "__main__":
    try:
        train_crop_recommender()
        train_yield_predictor()
        # This will now look for your 'train' folder
        train_disease_model(dataset_path='train') 
        print("\nAll models trained and saved successfully.")
    except Exception as e:
        print(f"\nError: {e}")