import os
from pathlib import Path

import kaggle


def main(config: dict) -> None:

    download_dataset_if_not_present(config["dataset"], config["download_dir_path"])


def download_dataset_if_not_present(dataset: str, download_dir_path: str) -> None:

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


if __name__ == "__main__":

    DATASET = "clmentbisaillon/fake-and-real-news-dataset"
    DOWNLOAD_TO_DIR = "data/input/kaggle"

    config = {
        "dataset": DATASET,
        "download_dir_path": DOWNLOAD_TO_DIR,
    }

    main(config)
