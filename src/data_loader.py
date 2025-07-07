import os

import kaggle
import pandas as pd
from pathlib import Path


def download_dataset_if_not_present(
    dataset: str,
    download_dir_path: str,
) -> None:
    """
    Download the Kaggle dataset if it's not already present.
    """
    true_news_data_path = Path(os.path.join(download_dir_path, "True.csv"))
    fake_news_data_path = Path(os.path.join(download_dir_path, "Fake.csv"))

    if true_news_data_path.exists() and fake_news_data_path.exists():
        print("Found existing dataset; skipping data download.")
        return

    print(f"Downloading dataset '{dataset}' to {download_dir_path}...")
    kaggle.api.dataset_download_files(
        dataset,
        path=download_dir_path,
        unzip=True,
    )
    print("Dataset downloaded and unzipped successfully.")


def load_dataframe(download_dir_path: str) -> pd.DataFrame:
    """
    Load True and Fake news CSVs, assign labels, and concatenate them.
    """
    print(f"Loading data from {download_dir_path}...")
    true_news_data = pd.read_csv(os.path.join(download_dir_path, "True.csv"))
    true_news_data["label"] = 0  # Assign 0 for True News

    fake_news_data = pd.read_csv(os.path.join(download_dir_path, "Fake.csv"))
    fake_news_data["label"] = 1  # Assign 1 for Fake News

    combined_data = pd.concat(
        [true_news_data, fake_news_data],
        ignore_index=True,
    )
    print(f"Loaded {len(combined_data)} total news articles.")
    return combined_data
