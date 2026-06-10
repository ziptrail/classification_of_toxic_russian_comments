import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, LSTM, GRU, Bidirectional,
                                     Conv1D, GlobalMaxPooling1D,
                                     Dense, Dropout, BatchNormalization, Input)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


app = Flask(__name__)

with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

max_words = config['max_words']
max_len = config['max_len']
label_cols = config['label_cols']

with open('tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_json = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_json)

def build_lstm():
    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(max_words, 300, trainable=False),
        LSTM(128, return_sequences=True),
        Dropout(0.4),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(label_cols), activation='sigmoid')
    ])
    return model

def build_gru():
    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(max_words, 300, trainable=False),
        Bidirectional(GRU(128, return_sequences=True)),
        Dropout(0.4),
        BatchNormalization(),
        Bidirectional(GRU(64, return_sequences=False)),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(label_cols), activation='sigmoid')
    ])
    return model

def build_conv1d():
    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(max_words, 300, trainable=False),
        Conv1D(128, 5, activation='relu', padding='same'),
        Dropout(0.4),
        BatchNormalization(),
        Conv1D(64, 3, activation='relu', padding='same'),
        GlobalMaxPooling1D(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(label_cols), activation='sigmoid')
    ])
    return model

models = {}

try:
    lstm = build_lstm()
    lstm.load_weights('model/lstm_toxic_comment_model_russian.h5')
    models['lstm'] = lstm
    print("Модель LSTM успешно загружена!")
except Exception as e:
    print(f"Ошибка загрузки LSTM: {e}")

try:
    gru = build_gru()
    gru.load_weights('model/gru_toxic_comment_model_russian.h5')
    models['gru'] = gru
    print("Модель GRU успешно загружена!")
except Exception as e:
    print(f"Ошибка загрузки GRU: {e}")

try:
    conv1d = build_conv1d()
    conv1d.load_weights('model/conv1d_toxic_comment_model_russian.h5')
    models['conv1d'] = conv1d
    print("Модель Conv1D успешно загружена!")
except Exception as e:
    print(f"Ошибка загрузки Conv1D: {e}")

def predict_text(model_name, text):
    model = models[model_name]
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded, verbose=0)[0]
    result = {}
    for label, prob in zip(label_cols, pred):
        result[label] = round(float(prob) * 100, 2)
    main_class = label_cols[np.argmax(pred)]
    main_conf = result[main_class]
    return main_class, main_conf, result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_ajax', methods=['POST'])
def predict_ajax():
    data = request.get_json()
    text = data.get('text', '').strip()
    model_name = data.get('model', 'lstm')

    if not text:
        return jsonify({'error': 'Введите текст'}), 400
    if model_name not in models:
        return jsonify({'error': f'Модель {model_name} не загружена'}), 400

    try:
        main_class, main_conf, details = predict_text(model_name, text)
        return jsonify({
            'prediction': main_class,
            'confidence': main_conf,
            'details': details,
            'text': text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)