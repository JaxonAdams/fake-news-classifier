import os
from pathlib import Path

import kaggle
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
    news_data = remove_unneeded_features(news_data)

    lowercaser = TextLowercaser()
    news_data["content"] = lowercaser.transform(news_data["content"])

    print("\nData sample:")
    print(news_data.sample(n=8))


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
    true_news_data["label"] = "0"

    fake_news_data = pd.read_csv(os.path.join(download_dir_path, "Fake.csv"))
    fake_news_data["label"] = "1"

    return pd.concat([true_news_data, fake_news_data])


def remove_unneeded_features(data: pd.DataFrame) -> pd.DataFrame:

    data["content"] = data["title"] + "[SEP]" + data["text"]
    return data.drop(columns=["subject", "date", "title", "text"])


if __name__ == "__main__":

    DATASET = "clmentbisaillon/fake-and-real-news-dataset"
    DOWNLOAD_TO_DIR = "data/input/kaggle"

    config = {
        "dataset": DATASET,
        "download_dir_path": DOWNLOAD_TO_DIR,
    }

    main(config)
