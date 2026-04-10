"""
COMP 475 / ACMP 466 - Practical Task 2
LSTM for Stock Trend Forecasting
Author: [Your Name]
Reg No: INP12.0004.24
"""

# ============================================================
# IMPORT ALL REQUIRED LIBRARIES
# ============================================================

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import time

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("=" * 70)
print("LSTM FOR STOCK TREND FORECASTING")
print("=" * 70)

# ============================================================
# STEP 1: DOWNLOAD STOCK DATA
# ============================================================

print("\n[STEP 1] Downloading stock data...")

ticker = "AAPL"  # Apple Inc. stock
start_date = "2015-01-01"
end_date = "2024-12-31"

data = yf.download(ticker, start=start_date, end=end_date, progress=False)

print(f"Downloaded {len(data)} days of data")
print(f"Date range: {start_date} to {end_date}")
print(f"Ticker: {ticker}")

# Display first few rows
print("\nFirst 5 rows of data:")
print(data.head())

# Extract closing prices
prices = data['Close'].values
dates = data.index

# ============================================================
# STEP 2: VISUALIZE HISTORICAL PRICES
# ============================================================

print("\n[STEP 2] Visualizing historical prices...")

plt.figure(figsize=(14, 6))
plt.plot(dates, prices, 'b-', linewidth=1, label=f'{ticker} Closing Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title(f'{ticker} Historical Closing Prices ({start_date} to {end_date})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('stock_prices.png', dpi=150)
plt.show()

print(f"\nPrice Statistics:")
print(f"  Minimum Price: ${prices.min():.2f}")
print(f"  Maximum Price: ${prices.max():.2f}")
print(f"  Average Price: ${prices.mean():.2f}")
print(f"  Standard Deviation: ${prices.std():.2f}")

# ============================================================
# STEP 3: CREATE SEQUENCES AND BINARY LABELS
# ============================================================

print("\n[STEP 3] Creating sequences and binary labels...")

SEQUENCE_LENGTH = 15  # Use 15 days of history to predict next day

X = []  # Input sequences
y = []  # Labels (1 = price up, 0 = price down)

for i in range(len(prices) - SEQUENCE_LENGTH - 1):
    # Input: sequence of closing prices
    X.append(prices[i:i + SEQUENCE_LENGTH])
    
    # Label: compare next day's price with current day
    next_day_price = prices[i + SEQUENCE_LENGTH + 1]
    current_day_price = prices[i + SEQUENCE_LENGTH]
    
    if next_day_price > current_day_price:
        y.append(1)  # Price will go UP
    else:
        y.append(0)  # Price will go DOWN

X = np.array(X)
y = np.array(y)

print(f"Total sequences created: {len(X)}")
print(f"Sequence shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Check class balance
up_count = np.sum(y)
down_count = len(y) - up_count
print(f"\nClass Distribution:")
print(f"  UP (1):   {up_count} ({up_count/len(y)*100:.1f}%)")
print(f"  DOWN (0): {down_count} ({down_count/len(y)*100:.1f}%)")

# ============================================================
# STEP 4: NORMALIZE THE DATA
# ============================================================

print("\n[STEP 4] Normalizing data...")

scaler = MinMaxScaler()
X_reshaped = X.reshape(-1, SEQUENCE_LENGTH)
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(-1, SEQUENCE_LENGTH, 1)

print(f"Normalized data shape: {X.shape}")
print(f"Data range after normalization: [{X.min():.2f}, {X.max():.2f}]")

# ============================================================
# STEP 5: SPLIT DATA INTO TRAIN/VALIDATION/TEST SETS
# ============================================================

print("\n[STEP 5] Splitting data into train/validation/test sets...")
print("Split ratio: 70% train, 15% validation, 15% test")

# First split: separate test set (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, shuffle=False, random_state=42
)

# Second split: separate validation from remaining (15% of original)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, shuffle=False, random_state=42
)

print(f"Training set:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test set:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# ============================================================
# STEP 6: BUILD THE LSTM MODEL
# ============================================================

print("\n[STEP 6] Building LSTM model...")

model = Sequential([
    # First LSTM layer (returns sequences for next layer)
    LSTM(units=64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)),
    Dropout(0.2),  # Prevents overfitting
    
    # Second LSTM layer
    LSTM(units=32, return_sequences=False),
    Dropout(0.2),  # Prevents overfitting
    
    # Dense layer for processing
    Dense(units=16, activation='relu'),
    
    # Output layer with sigmoid for binary classification
    Dense(units=1, activation='sigmoid')
])

# Display model architecture
model.summary()

# ============================================================
# STEP 7: COMPILE THE MODEL
# ============================================================

print("\n[STEP 7] Compiling model...")

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ============================================================
# STEP 8: TRAIN THE MODEL
# ============================================================

print("\n[STEP 8] Training model...")
print("-" * 50)

start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

# ============================================================
# STEP 9: EVALUATE ON TEST SET
# ============================================================

print("\n[STEP 9] Evaluating on test set...")

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# ============================================================
# STEP 10: MAKE PREDICTIONS
# ============================================================

print("\n[STEP 10] Making predictions on test set...")

y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# ============================================================
# STEP 11: BASELINE COMPARISON
# ============================================================

print("\n[STEP 11] Baseline comparison...")

# Baseline: always predict the majority class from training
majority_class = 1 if np.sum(y_train) > len(y_train) / 2 else 0
baseline_pred = np.full_like(y_test, majority_class)
baseline_accuracy = accuracy_score(y_test, baseline_pred)

print(f"LSTM Test Accuracy:     {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Baseline Accuracy:      {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"Improvement:            {test_accuracy - baseline_accuracy:.4f} ({(test_accuracy - baseline_accuracy)*100:.2f}%)")

if test_accuracy > baseline_accuracy + 0.03:
    print("\n✓ CONCLUSION: LSTM shows meaningful improvement over baseline.")
else:
    print("\n⚠️ CONCLUSION: LSTM is NOT significantly better than baseline.")

# ============================================================
# STEP 12: CONFUSION MATRIX
# ============================================================

print("\n[STEP 12] Generating confusion matrix...")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Reds',
    xticklabels=['DOWN (0)', 'UP (1)'],
    yticklabels=['DOWN (0)', 'UP (1)']
)
plt.title(f'Confusion Matrix - {ticker} Stock Direction Prediction', fontsize=14)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.savefig('confusion_matrix_lstm.png', dpi=150)
plt.show()

# ============================================================
# STEP 13: CLASSIFICATION REPORT
# ============================================================

print("\n[STEP 13] Classification Report:")
print("=" * 70)

print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))

# ============================================================
# STEP 14: PLOT TRAINING HISTORY
# ============================================================

print("\n[STEP 14] Plotting training history...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
ax1.plot(history.history['loss'], 'b-', linewidth=2, label='Training Loss')
ax1.plot(history.history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Curves')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy plot
ax2.plot(history.history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
ax2.plot(history.history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy Curves')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_lstm.png', dpi=150)
plt.show()

# ============================================================
# STEP 15: DISCUSSION AND ANALYSIS
# ============================================================

print("\n[STEP 15] Discussion and Analysis:")
print("=" * 70)

print("\n📊 PERFORMANCE SUMMARY:")
print(f"   LSTM Accuracy:     {test_accuracy:.4f}")
print(f"   Baseline Accuracy: {baseline_accuracy:.4f}")
print(f"   Improvement:       {test_accuracy - baseline_accuracy:.4f}")

print("\n📉 WHY STOCK PREDICTION IS CHALLENGING:")
print("   1. Efficient Market Hypothesis - prices reflect all known information")
print("   2. Random Walk Theory - price changes are largely unpredictable")
print("   3. External factors: news, earnings reports, geopolitical events")
print("   4. Investor sentiment and irrational behavior")
print("   5. Non-stationary distribution - patterns change over time")
print("   6. Low signal-to-noise ratio")
print("   7. Feedback loops - successful strategies get exploited away")

print("\n🔧 SUGGESTIONS FOR IMPROVEMENT:")
print("   1. Add technical indicators (RSI, MACD, Bollinger Bands)")
print("   2. Include volume and volatility features")
print("   3. Incorporate sentiment analysis from news/social media")
print("   4. Use Transformer or Attention mechanisms instead of LSTM")
print("   5. Ensemble multiple models for better predictions")
print("   6. Try different sequence lengths (10, 20, 30 days)")
print("   7. Add macroeconomic features (interest rates, inflation)")

# Save the model
model.save('lstm_stock_model.h5')
print("\n✅ Model saved as 'lstm_stock_model.h5'")

print("\n" + "=" * 70)
print("LSTM TASK COMPLETED SUCCESSFULLY!")
print("=" * 70)