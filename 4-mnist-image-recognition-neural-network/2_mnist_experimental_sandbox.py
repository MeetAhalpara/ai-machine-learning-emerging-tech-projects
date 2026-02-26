"""
MNIST Experimental Evaluation Environment
Author: Meet Ahalpara

This module serves as an advanced sandbox for the MNIST image classification pipeline. 
It introduces targeted data exploration (isolating specific digit classes) and implements 
visual validation tracking across the training epochs to diagnose underfitting or overfitting.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt

print("\n" + "="*60)
print(" MNIST EXPERIMENTAL SANDBOX INITIATED")
print("="*60)

print("\n[INFO] Ingesting MNIST handwritten digit dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"[SUCCESS] Data Loaded. Matrix Shape: {x_train.shape}")

# A custom spatial search is executed to isolate the first index instance of digits 0-9.
print("\n[INFO] Executing spatial search to isolate sequential digits (0-9)...")
plt.figure(figsize=(10, 3))

digit_indices = []
for digit in range(10): 
    idx = tf.where(y_train == digit)[0][0]  
    digit_indices.append(int(idx))

print(f"|-- Target Indices located at: {digit_indices}")

print("\n[INFO] Generating ordered visual preview (Close window to continue)...")
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[digit_indices[i]], cmap="gray")
    plt.title(f"Digit: {i}", fontweight='bold') 
    plt.axis("off")

plt.tight_layout()
plt.show()

print("\n[INFO] Normalizing mathematical pixel intensity arrays...")
x_train = x_train / 255.0
x_test = x_test / 255.0

print("\n[INFO] Compiling Experimental Neural Network architecture...")
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), 
    tf.keras.layers.Dense(128, activation='relu', name="HiddenLayer"), 
    tf.keras.layers.Dense(10, activation='softmax', name="OutputLayer") 
])

print("-" * 60)
model.summary()
print("-" * 60)

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

print("\n[INFO] Initiating monitored model training...")
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1) 

print("\n[INFO] Rendering historical convergence trajectories (Close window to continue)...")
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Convergence Tracking', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n[INFO] Evaluating model performance...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"[SUCCESS] Sandbox Model Test Accuracy: {test_acc * 100:.2f}%")

print("\n" + "="*60)
print(" EXPERIMENT: ACTIVATION FUNCTION IMPACT (SIGMOID)")
print("="*60)

model_sigmoid = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='sigmoid'),   
    tf.keras.layers.Dense(10, activation='softmax')
])

model_sigmoid.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n[INFO] Training experimental Sigmoid model...")
history_sigmoid = model_sigmoid.fit(x_train, y_train, epochs=5, validation_split=0.1)

test_loss_sigmoid, test_acc_sigmoid = model_sigmoid.evaluate(x_test, y_test)

print("\n" + "="*60)
print(" EXECUTIVE SUMMARY: SANDBOX RESULTS")
print("="*60)
print(f"|-- Standard ReLU Architecture Accuracy : {test_acc * 100:.2f}%")
print(f"|-- Legacy Sigmoid Architecture Accuracy: {test_acc_sigmoid * 100:.2f}%")
print("="*60 + "\n")