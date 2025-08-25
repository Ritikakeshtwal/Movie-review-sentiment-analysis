import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten, Dropout
from sklearn.metrics import confusion_matrix

# ===============================
# 1. Create "outputs" folder
# ===============================
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# ===============================
# 2. Load IMDB Dataset
# ===============================
vocab_size = 10000   # top 10k words
max_len = 200        # max review length

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences to same length
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# ===============================
# 3. Build ANN Model with Embedding
# ===============================
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_len),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # binary output
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ===============================
# 4. Train Model
# ===============================
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=128,
                    validation_split=0.2,
                    verbose=1)

# ===============================
# 5. Evaluate
# ===============================
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("Test Accuracy:", acc)

# ===============================
# 6. Save Accuracy & Loss Graphs
# ===============================
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.savefig("outputs/accuracy.png")
plt.show()
plt.close()

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")
plt.savefig("outputs/loss.png")
plt.show()
plt.close()

# ===============================
# 7. Confusion Matrix
# ===============================
y_pred = (model.predict(x_test) > 0.5).astype("int32")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Negative","Positive"], yticklabels=["Negative","Positive"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig("outputs/confusion_matrix.png")
plt.show()
plt.close()

# ===============================
# 8. Try Some Custom Predictions
# ===============================
word_index = imdb.get_word_index()
reverse_word_index = {v+3: k for k, v in word_index.items()}
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i, '?') for i in encoded_review])

# Example review check
print("\nSample review (decoded):")
print(decode_review(x_test[0]))
print("Prediction:", "Positive" if y_pred[0][0] == 1 else "Negative")

print("✅ All outputs saved in 'outputs/' folder")