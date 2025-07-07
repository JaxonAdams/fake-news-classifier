import os

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from config import config


def load_or_train_model(
    model_path: str,
    vocab_size: int,
    embedding_dim: int,
    maxlen: int,
    X_train,
    y_train,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History | None]:
    """
    Load an existing model if available, otherwise compile and train a new one.
    Return the model and the history object (or None if loaded).
    """
    history = None

    if os.path.exists(model_path):
        print(f"\nLoading model from {model_path}...")
        model = load_model(model_path)
        print("Model loaded successfully.")
    else:
        print("\nModel not found. Training a new model...")
        model = Sequential()
        model.add(
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=maxlen,
            )
        )
        model.add(LSTM(units=config["lstm_units"], return_sequences=False))
        model.add(Dropout(config["dropout_rate"]))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        print("Training the model...")
        history = model.fit(
            X_train,
            y_train,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            validation_split=config["validation_split"],
            verbose=1,
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"Model saved to {model_path}")
    return model, history
