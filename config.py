import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

config = {
    "dataset": "clmentbisaillon/fake-and-real-news-dataset",
    "download_dir_path": os.path.join(DATA_DIR, "input", "kaggle"),
    "model_save_path": os.path.join(DATA_DIR, "output", "model", "LSTM.keras"),
    "visualizations_save_path": os.path.join(DATA_DIR, "output", "visualizations"),
    "tokenizer_num_words": 10000,
    "embedding_dim": 50,
    "lstm_units": 100,
    "dropout_rate": 0.5,
    "lstm_units": 100,
    "dropout_rate": 0.5,
    "epochs": 20,
    "batch_size": 32,
    "validation_split": 0.1,
    "test_size": 0.2,
    "random_state": 42,
    "prediction_threshold": 0.5,
}
