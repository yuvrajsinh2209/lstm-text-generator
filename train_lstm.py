import numpy as np
import tensorflow as tf
import string
import random

# ==============================
# 1. LOAD & PREPROCESS TEXT
# ==============================

# Load text file
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# Remove punctuation
text = text.translate(str.maketrans("", "", string.punctuation))

print("Text length:", len(text))

# Get unique characters
chars = sorted(list(set(text)))
print("Unique characters:", len(chars))

# Create mapping
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# ==============================
# 2. CREATE INPUT SEQUENCES
# ==============================

seq_length = 40
step = 3

sequences = []
next_chars = []

for i in range(0, len(text) - seq_length, step):
    sequences.append(text[i:i + seq_length])
    next_chars.append(text[i + seq_length])

print("Number of sequences:", len(sequences))

# Vectorization
X = np.zeros((len(sequences), seq_length, len(chars)), dtype=np.bool_)
y = np.zeros((len(sequences), len(chars)), dtype=np.bool_)

for i, seq in enumerate(sequences):
    for t, char in enumerate(seq):
        X[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

# ==============================
# 3. BUILD LSTM MODEL
# ==============================

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(seq_length, len(chars))),
    tf.keras.layers.Dense(len(chars), activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam"
)

model.summary()

# ==============================
# 4. TRAIN MODEL
# ==============================

model.fit(
    X,
    y,
    batch_size=128,
    epochs=10
)

# ==============================
# 5. TEXT GENERATION
# ==============================

def generate_text(seed, length=300):
    generated = seed
    sentence = seed[-seq_length:]

    for _ in range(length):
        x_pred = np.zeros((1, seq_length, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_idx[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = idx_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    return generated

# Random seed from text
start_index = random.randint(0, len(text) - seq_length - 1)
seed_text = text[start_index:start_index + seq_length]

print("\nSeed text:")
print(seed_text)

print("\nGenerated text:")
print(generate_text(seed_text))
