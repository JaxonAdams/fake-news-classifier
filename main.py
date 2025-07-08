import os

import nltk
import boto3
from sklearn.model_selection import train_test_split

from src.data_loader import download_dataset_if_not_present, load_dataframe
from src.preprocessing import (
    remove_unneeded_features,
    tokenize_and_pad_text,
    save_tokenizer,
)
from src.model import load_or_train_model
from src.visualizations import (
    plot_headline_length_distribution,
    plot_top_words,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_model_performance,
)
from src.custom_transformers import TextLowercaser
from config import config


# Ensure stopwords are downloaded
try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords")


def main() -> None:

    download_dir_path = config["download_dir_path"]
    model_save_path = config["model_save_path"]
    tokenizer_save_path = config["tokenizer_save_path"]
    visualizations_save_path = config["visualizations_save_path"]
    model_assets_dir = os.path.dirname(model_save_path)

    # Create output directories if they don't exist
    os.makedirs(download_dir_path, exist_ok=True)
    os.makedirs(model_assets_dir, exist_ok=True)
    os.makedirs(visualizations_save_path, exist_ok=True)

    # Step 1: Load Data
    download_dataset_if_not_present(config["dataset"], download_dir_path)
    news_data = load_dataframe(download_dir_path)

    # Step 2: Preprocessing and Data Exploration Visualizations
    print("Cleaning the dataset...")
    # Apply lowercasing and feature removal early for consistent data
    lowercaser = TextLowercaser()
    news_data["content"] = lowercaser.transform(news_data["title"])
    news_data_cleaned = remove_unneeded_features(news_data.copy())

    # Data Exploration Visualizations (using news_data_cleaned)
    plot_headline_length_distribution(
        news_data_cleaned,
        visualizations_save_path,
    )
    plot_top_words(news_data_cleaned, visualizations_save_path)

    # Tokenize and pad for model training
    padded_sequences, tokenizer, maxlen = tokenize_and_pad_text(news_data_cleaned)
    save_tokenizer(tokenizer, tokenizer_save_path)
    print(f"\nMAX SEQUENCE LENGTH: {maxlen}")

    # Step 3: Split train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences,
        news_data_cleaned["label"],
        test_size=0.2,
        random_state=42,
    )

    # Step 4: Compile, train, and test the model
    embedding_dim = 50
    vocab_size = len(tokenizer.word_index) + 1

    model, history = load_or_train_model(
        model_save_path,
        vocab_size,
        embedding_dim,
        maxlen,
        X_train,
        y_train,
    )

    # Evaluate model on test set
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Concatenated Model Test Loss: {loss:.4f}")
    print(f"Concatenated Model Test Accuracy: {accuracy:.4f}")

    # Generate model performance visualizations
    y_pred_proba = model.predict(X_test)
    plot_model_performance(history, visualizations_save_path)
    plot_confusion_matrix(y_test, y_pred_proba, visualizations_save_path)
    plot_roc_curve(y_test, y_pred_proba, visualizations_save_path)

    # Step 5: Upload model and tokenizer to S3 for Lambda deployment
    s3 = boto3.client("s3")
    bucket_name = config["s3_bucket_name"]
    model_s3_key = config["s3_model_key"]
    tokenizer_s3_key = config["s3_tokenizer_key"]

    print(
        f"\nUploading model '{os.path.basename(model_save_path)}' to s3://{bucket_name}/{model_s3_key}..."
    )
    s3.upload_file(model_save_path, bucket_name, model_s3_key)
    print("Model uploaded.")

    print(
        f"Uploading tokenizer '{os.path.basename(tokenizer_save_path)}' to s3://{bucket_name}/{tokenizer_s3_key}..."
    )
    s3.upload_file(tokenizer_save_path, bucket_name, tokenizer_s3_key)
    print("Tokenizer uploaded.")


if __name__ == "__main__":
    main()
