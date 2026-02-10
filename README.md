# LSTM Text Generator

This project implements a character-level text generation model using an LSTM (Long Short-Term Memory) neural network.

## Overview
The model is trained on a large public-domain text dataset (Shakespeare’s Complete Works) and learns character-level patterns to generate new text based on a seed input.

## Dataset
- Project Gutenberg – Shakespeare Complete Works  
- https://www.gutenberg.org/files/100/100-0.txt

## Model Architecture
- Character-level tokenization
- LSTM layer with 128 units
- Dense output layer with softmax activation
- Loss function: Categorical Crossentropy
- Optimizer: Adam
- Early stopping used to prevent overfitting

## How to Run
```bash
pip install -r requirements.txt
python train_lstm.py
