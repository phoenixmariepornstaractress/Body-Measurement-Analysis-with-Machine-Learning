# Deep Learning Model Integration for Body Measurement Prediction

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset (assumed to be preprocessed and cleaned)
data = pd.read_csv("body_measurements.csv")

# Define features and target
FEATURE_COLUMNS = ["bust", "waist"]
TARGET_COLUMN = "hips"

X = data[FEATURE_COLUMNS].values
y = data[TARGET_COLUMN].values

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to build the neural network model
def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Instantiate and compile the model
model = build_model(X_train_scaled.shape[1])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate the trained model
def evaluate_model(model, X_test, y_test):
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}, Test MAE: {mae:.4f}")
    return loss, mae

evaluate_model(model, X_test_scaled, y_test)

# Generate and print predictions
def show_predictions(model, X_test):
    predictions = model.predict(X_test)
    print("\nSample Predictions:", predictions[:5].flatten())
    return predictions

predictions = show_predictions(model, X_test_scaled)

# Plot the training and validation loss over epochs
def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Save the trained model to disk
def save_model(model, filename="body_measurement_model.h5"):
    model.save(filename)
    print(f"Model saved to {filename}")

save_model(model)

# Load a trained model from disk
def load_trained_model(filename="body_measurement_model.h5"):
    return tf.keras.models.load_model(filename)

# Example usage
# loaded_model = load_trained_model()
# evaluate_model(loaded_model, X_test_scaled, y_test)
