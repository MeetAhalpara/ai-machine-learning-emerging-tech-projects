"""
MNIST Image Recognition Neural Network Pipeline
Author: Meet Ahalpara

This module implements a foundational Sequential Neural Network using TensorFlow 
and Keras. It is designed to classify 28x28 pixel grayscale images of handwritten 
digits. The pipeline encompasses data ingestion, normalization, network construction, 
and an experimental comparison of architectural activation functions.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses ambient TensorFlow framework warnings
import tensorflow as tf
import matplotlib.pyplot as plt

print("\n" + "="*60)
print(" MNIST IMAGE RECOGNITION PIPELINE INITIATED")
print("="*60)

# ==========================================
# PHASE 1: DATA INGESTION & EXPLORATION
# ==========================================
print("\n[INFO] Ingesting MNIST handwritten digit dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("[SUCCESS] Dataset successfully loaded into memory.")
print(f"|-- Training Samples : {len(x_train)}")
print(f"|-- Testing Samples  : {len(x_test)}")
print(f"|-- Image Resolution : {x_train[0].shape[0]}x{x_train[0].shape[1]} Pixels")

# A visual sample of the raw dataset is generated to ensure extraction integrity.
print("\n[INFO] Generating structural visual preview (Close window to continue)...")
plt.figure(figsize=(10, 3))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# ==========================================
# PHASE 2: DATA NORMALIZATION
# ==========================================
# Raw pixel intensities range from 0 to 255. These values are mathematically scaled 
# down to a floating-point range between 0.0 and 1.0. This normalization accelerates 
# convergence and prevents numerical overflow during backpropagation.
print("\n[INFO] Normalizing pixel intensity matrices...")
x_train = x_train / 255.0
x_test = x_test / 255.0

print(f"|-- Maximum pixel value post-scaling: {x_train.max():.1f}")
print(f"|-- Minimum pixel value post-scaling: {x_train.min():.1f}")

# ==========================================
# PHASE 3: NEURAL NETWORK ARCHITECTURE
# ==========================================
# A Sequential Multi-Layer Perceptron (MLP) is constructed. 
print("\n[INFO] Compiling Sequential Neural Network architecture...")
model = tf.keras.Sequential([
    # Input Layer: Flattens the 2D 28x28 matrices into 1D 784-element vectors.
    tf.keras.layers.Flatten(input_shape=(28, 28)),  
    # Hidden Layer: 128 nodes utilizing a Rectified Linear Unit (ReLU) activation to learn non-linear patterns.
    tf.keras.layers.Dense(128, activation='relu', name="HiddenLayer"), 
    # Output Layer: 10 nodes (one per digit class) utilizing Softmax to output a probability distribution.
    tf.keras.layers.Dense(10, activation='softmax', name="OutputLayer") 
])

print("-" * 60)
model.summary()
print("-" * 60)

# ==========================================
# PHASE 4: COMPILATION & TRAINING
# ==========================================
model.compile(
    optimizer='adam', # Adam optimizer dynamically adapts the learning rate to accelerate convergence.
    loss='sparse_categorical_crossentropy',  # Ideal loss function for mutually exclusive integer labels.
    metrics=['accuracy']
)

print("\n[INFO] Initiating model training over 5 epochs...")
# 10% of the training data is isolated for validation, ensuring the model's capacity 
# to generalize to unseen data is tracked during the learning process.
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# ==========================================
# PHASE 5: SYSTEM EVALUATION
# ==========================================
print("\n[INFO] Evaluating baseline model against the testing partition...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"[SUCCESS] Baseline Model Test Accuracy: {test_acc * 100:.2f}%")

# ==========================================
# PHASE 6: ARCHITECTURAL EXPERIMENTATION
# ==========================================
# Evaluates the structural impact of replacing the modern ReLU activation 
# with a traditional Sigmoid function in the hidden layer.
print("\n" + "="*60)
print(" EXPERIMENT: ACTIVATION FUNCTION IMPACT (SIGMOID)")
print("="*60)

model_sigmoid = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='sigmoid'),   # Altered from ReLU
    tf.keras.layers.Dense(10, activation='softmax')     # NOTE: The extra Dense layer was removed to match baseline complexity
])

model_sigmoid.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n[INFO] Training experimental Sigmoid model...")
history_sigmoid = model_sigmoid.fit(x_train, y_train, epochs=5, validation_split=0.1)

print("\n[INFO] Evaluating experimental model against the testing partition...")
test_loss_sigmoid, test_acc_sigmoid = model_sigmoid.evaluate(x_test, y_test)

# Final analytical comparison
print("\n" + "="*60)
print(" EXECUTIVE SUMMARY: EXPERIMENT RESULTS")
print("="*60)
print(f"|-- Standard ReLU Architecture Accuracy : {test_acc * 100:.2f}%")
print(f"|-- Legacy Sigmoid Architecture Accuracy: {test_acc_sigmoid * 100:.2f}%")
print("="*60 + "\n")