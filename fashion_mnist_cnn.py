"""
COMP 475 / ACMP 466 - Practical Task 1
CNN for Fashion-MNIST Classification
Author: [Your Name]
Reg No: INP12.0004.24
"""

# ============================================================
# IMPORT ALL REQUIRED LIBRARIES
# ============================================================

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("=" * 70)
print("CNN FOR FASHION-MNIST CLASSIFICATION")
print("=" * 70)

# ============================================================
# STEP 1: LOAD THE DATASET
# ============================================================

print("\n[STEP 1] Loading Fashion-MNIST dataset...")

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Class names for Fashion-MNIST
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

print(f"Training images: {x_train.shape}")
print(f"Training labels: {y_train.shape}")
print(f"Test images: {x_test.shape}")
print(f"Test labels: {y_test.shape}")
print(f"Number of classes: {len(class_names)}")

# Display sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i], cmap='gray')
    ax.set_title(class_names[y_train[i]])
    ax.axis('off')
plt.suptitle("Sample Fashion-MNIST Images", fontsize=14)
plt.savefig('sample_images.png')
plt.show()

# ============================================================
# STEP 2: PREPROCESS THE DATA
# ============================================================

print("\n[STEP 2] Preprocessing data...")

# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Add channel dimension (28, 28, 1) for CNN
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print(f"Training shape after preprocessing: {x_train.shape}")
print(f"Test shape after preprocessing: {x_test.shape}")

# ============================================================
# STEP 3: BUILD THE CNN MODEL
# ============================================================

print("\n[STEP 3] Building CNN model...")

model = keras.Sequential([
    # First convolutional layer
    keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        input_shape=(28, 28, 1)
    ),
    # First max pooling layer
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Second convolutional layer
    keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        padding='same'
    ),
    # Second max pooling layer
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten layer to convert 2D to 1D
    keras.layers.Flatten(),
    
    # Fully connected layer
    keras.layers.Dense(128, activation='relu'),
    
    # Dropout for regularization (prevents overfitting)
    keras.layers.Dropout(0.5),
    
    # Output layer with 10 classes
    keras.layers.Dense(10, activation='softmax')
])

# Display model architecture
model.summary()

# ============================================================
# STEP 4: COMPILE THE MODEL
# ============================================================

print("\n[STEP 4] Compiling model...")

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# STEP 5: TRAIN THE MODEL
# ============================================================

print("\n[STEP 5] Training model...")
print("-" * 50)

start_time = time.time()

history = model.fit(
    x_train, y_train,
    validation_split=0.2,  # Use 20% for validation
    epochs=12,
    batch_size=32,
    verbose=1
)

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

# ============================================================
# STEP 6: EVALUATE ON TEST SET
# ============================================================

print("\n[STEP 6] Evaluating on test set...")

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# ============================================================
# STEP 7: PLOT ACCURACY AND LOSS CURVES
# ============================================================

print("\n[STEP 7] Generating accuracy and loss curves...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(history.history['accuracy'], 'b-', linewidth=2, label='Training')
ax1.plot(history.history['val_accuracy'], 'r-', linewidth=2, label='Validation')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Training vs Validation Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss plot
ax2.plot(history.history['loss'], 'b-', linewidth=2, label='Training')
ax2.plot(history.history['val_loss'], 'r-', linewidth=2, label='Validation')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training vs Validation Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('accuracy_loss_curves.png', dpi=150)
plt.show()

# ============================================================
# STEP 8: CONFUSION MATRIX
# ============================================================

print("\n[STEP 8] Generating confusion matrix...")

# Make predictions
y_pred = np.argmax(model.predict(x_test), axis=1)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title('Confusion Matrix - Fashion-MNIST', fontsize=14)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# ============================================================
# STEP 9: CLASSIFICATION REPORT
# ============================================================

print("\n[STEP 9] Classification Report:")
print("=" * 70)

report = classification_report(y_test, y_pred, target_names=class_names)
print(report)

# ============================================================
# STEP 10: ANALYSIS AND DISCUSSION
# ============================================================

print("\n[STEP 10] Analysis and Discussion:")
print("=" * 70)

final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
overfitting_gap = final_train_acc - final_val_acc

print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Overfitting Gap: {overfitting_gap:.4f}")

if overfitting_gap > 0.05:
    print("\n⚠️ OBSERVATION: The model shows signs of overfitting.")
    print("   Training accuracy is significantly higher than validation accuracy.")
else:
    print("\n✓ OBSERVATION: The model generalizes well.")

print("\n📊 MOST CONFUSED CLASSES:")
print("   1. Shirt ↔ Pullover (similar loose-fitting clothing)")
print("   2. Coat ↔ Pullover (outerwear categories)")
print("   3. Shirt ↔ T-shirt/top (similar tops)")

print("\n🔧 SUGGESTIONS FOR IMPROVEMENT:")
print("   1. Add data augmentation (rotation, zoom, shift)")
print("   2. Add Batch Normalization layers")
print("   3. Increase dropout rate to 0.6")
print("   4. Add more convolutional layers (128 filters)")
print("   5. Use learning rate scheduling")
print("   6. Implement early stopping")

# Save the model
model.save('fashion_mnist_cnn_model.h5')
print("\n✅ Model saved as 'fashion_mnist_cnn_model.h5'")

print("\n" + "=" * 70)
print("CNN TASK COMPLETED SUCCESSFULLY!")
print("=" * 70)