import os
from pathlib import Path

import kaggle
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


# Custom Transformer for lowercasing
class TextLowercaser(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # Nothing to learn in this step

    def transform(self, X):
        return [text.lower() for text in X]


def main(config: dict) -> None:

    download_dir_path = config["download_dir_path"]
    model_save_path = config["model_save_path"]
    visualizations_save_path = config["visualizations_save_path"]

    # Step 1: Load
    download_dataset_if_not_present(config["dataset"], download_dir_path)
    news_data = load_dataframe(download_dir_path)

    # Step 2: Clean
    print("Cleaning the dataset...")
    news_data = remove_unneeded_features(news_data)

    lowercaser = TextLowercaser()
    news_data["content"] = lowercaser.transform(news_data["content"])

    padded_sequences, tokenizer, maxlen = tokenize_and_pad_text(news_data)

    # Step 3: Split train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences,
        news_data["label"],
        test_size=0.2,
        random_state=42,
    )

    # Step 4: Compile, train, and test the model
    embedding_dim = 50
    vocab_size = len(tokenizer.word_index) + 1

    model = load_or_train_model(
        model_save_path,
        vocab_size,
        embedding_dim,
        maxlen,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Concatenated Model Test Loss: {loss:.4f}")
    print(f"Concatenated Model Test Accuracy: {accuracy:.4f}")

    y_pred_proba = model.predict(X_test)
    plot_confusion_matrix(
        y_test,
        y_pred_proba,
        save_path=visualizations_save_path,
    )
    plot_roc_curve(y_test, y_pred_proba, save_path=visualizations_save_path)

    print("\n\nPREDICTING ON NEW DATA:")

    headlines = [
        "Trump dismisses Musk's political ambitions as 'ridiculous' in sharp rebuke",  # Fox News
        "Camp Mystic confirms 27 dead in Texas floods as more rain looms",  # CNN
        "After setback to Iran's nuclear program, Trump expected to leverage military support in Netanyahu meeting",  # Fox News
        "Peter Thiel Shows Trump How To Sort Spreadsheet Of Americans By Ethnicity",  # The Onion
        "Study Finds Tiny Nose Robots Can Be Used To Clean Sinuses",  # The Onion
    ]

    headlines_lower = lowercaser.transform(headlines)
    new_sequences = tokenizer.texts_to_sequences(headlines_lower)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=maxlen, padding="post")

    predictions = model.predict(new_padded_sequences)

    for i, headline in enumerate(headlines):
        is_fake = (
            "LIKELY FAKE!" if predictions[i][0] > 0.5 else "Not likely to be fake."
        )
        print(f'\nHeadline:\n"{headline}"\nModel prediction: "{is_fake}"')


def download_dataset_if_not_present(
    dataset: str,
    download_dir_path: str,
) -> None:

    true_news_data_path = Path(os.path.join(download_dir_path, "True.csv"))
    fake_news_data_path = Path(os.path.join(download_dir_path, "Fake.csv"))

    if true_news_data_path.exists() and fake_news_data_path.exists():
        print("Found existing dataset; skipping data download.")
        return

    kaggle.api.dataset_download_files(
        dataset,
        path=download_dir_path,
        unzip=True,
    )


def load_dataframe(download_dir_path: str) -> pd.DataFrame:

    true_news_data = pd.read_csv(os.path.join(download_dir_path, "True.csv"))
    true_news_data["label"] = 0

    fake_news_data = pd.read_csv(os.path.join(download_dir_path, "Fake.csv"))
    fake_news_data["label"] = 1

    return pd.concat([true_news_data, fake_news_data])


def remove_unneeded_features(data: pd.DataFrame) -> pd.DataFrame:

    data["content"] = data["title"]
    return data.drop(columns=["subject", "date", "title", "text"])


def tokenize_and_pad_text(data: pd.DataFrame):

    print("\nTokenizing data...")
    tokenizer = Tokenizer(num_words=1000, oov_token="<unk>")
    tokenizer.fit_on_texts(data["content"])
    word_index = tokenizer.word_index
    print(f"Found {len(word_index)} unique tokens in the dataset.")

    sequences = tokenizer.texts_to_sequences(data["content"])
    maxlen = max([len(seq) for seq in sequences])
    return (
        pad_sequences(sequences, maxlen=maxlen, padding="post"),
        tokenizer,
        maxlen,
    )


def load_or_train_model(
    model_path: str,
    vocab_size: int,
    embedding_dim: int,
    maxlen: int,
    X_train,
    y_train,
    X_test,
    y_test,
) -> tf.keras.Model:
    """
    Loads an existing model if available, otherwise compiles and trains a new one.
    """
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
        model.add(LSTM(units=100, return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        print("Training the concatenated model...")
        model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=2,
            validation_split=0.1,
            verbose=1,
        )
        # Save the trained model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"Model saved to {model_path}")
    return model


def plot_confusion_matrix(y_true, y_pred_proba, save_path, threshold=0.5):

    y_pred_classes = (y_pred_proba > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Real News (0)", "Fake News (1)"],
        yticklabels=["Real News (0)", "Fake News (1)"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))


def plot_roc_curve(y_true, y_pred_proba, save_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.2f})",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        color="navy",
        lw=2,
        linestyle="--",
        label="Random Classifier",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "roc_curve.png"))


if __name__ == "__main__":

    DATASET = "clmentbisaillon/fake-and-real-news-dataset"
    DOWNLOAD_TO_DIR = "data/input/kaggle"
    MODEL_DIR = "data/output/model/LSTM.keras"
    VISUALIZATIONS_DIR = "data/output/visualizations"

    config = {
        "dataset": DATASET,
        "download_dir_path": DOWNLOAD_TO_DIR,
        "model_save_path": MODEL_DIR,
        "visualizations_save_path": VISUALIZATIONS_DIR,
    }

    main(config)
