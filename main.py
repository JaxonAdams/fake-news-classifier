import os
from pathlib import Path

import kaggle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.models import Sequential
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

    print("\nTraining the concatenated model...")
    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=2,
        validation_split=0.1,
        verbose=0,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Concatenated Model Test Loss: {loss:.4f}")
    print(f"Concatenated Model Test Accuracy: {accuracy:.4f}")


def download_dataset_if_not_present(
    dataset: str,
    download_dir_path: str,
) -> None:

    true_news_data_path = Path(os.path.join(download_dir_path, "True.csv"))
    fake_news_data_path = Path(os.path.join(download_dir_path, "Fake.csv"))

    if true_news_data_path.exists and fake_news_data_path.exists():
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

    data["content"] = data["title"] + "[SEP]" + data["text"]
    return data.drop(columns=["subject", "date", "title", "text"])


def tokenize_and_pad_text(data: pd.DataFrame):

    print("\nTokenizing data...")
    tokenizer = Tokenizer(num_words=100, oov_token="<unk>")
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


if __name__ == "__main__":

    DATASET = "clmentbisaillon/fake-and-real-news-dataset"
    DOWNLOAD_TO_DIR = "data/input/kaggle"

    config = {
        "dataset": DATASET,
        "download_dir_path": DOWNLOAD_TO_DIR,
    }

    main(config)
