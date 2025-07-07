import os
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report


def plot_model_performance(
    history: tf.keras.callbacks.History | None, save_path: str
) -> None:
    """
    Plot the training and validation accuracy and loss from the model history.
    Save the plot to the specified path.
    """
    if history is None:
        print("\nNo training history available to plot (model was loaded).")
        return

    print("\nGenerating performance plots...")

    # Accuracy plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_file = os.path.join(save_path, "training_validation_performance.png")
    plt.savefig(plot_file)
    print(f"Saved training/validation performance plot to {plot_file}")
    plt.close()


def plot_headline_length_distribution(news_data: pd.DataFrame, save_path: str):
    """
    Plot the distribution of headline lengths for real and fake news.
    Save the plot to the specified path.
    """
    news_data["content_length"] = news_data["content"].apply(lambda x: len(x.split()))

    news_data["label_name"] = news_data["label"].map({0: "Real News", 1: "Fake News"})

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=news_data,
        x="content_length",
        hue="label_name",
        kde=True,
        bins=50,
        palette="viridis",
        hue_order=["Real News", "Fake News"],
    )
    plt.title("Distribution of Headline Lengths by News Type")
    plt.xlabel("Number of Words in Headline")
    plt.ylabel("Frequency")
    legend = plt.gca().get_legend()
    if legend:
        legend.set_title("News Type")

    plot_file = os.path.join(save_path, "headline_length_distribution.png")
    plt.savefig(plot_file)
    print(f"Saved headline length distribution plot to {plot_file}")
    plt.close()

    print("\nHeadline Length Statistics:")
    print(news_data.groupby("label")["content_length"].describe())


def plot_top_words(news_data: pd.DataFrame, save_path: str, top_n: int = 20):
    """
    Plot the most common words in real and fake news headlines.
    Save the plot to the specified path.
    """
    stop_words = set(stopwords.words("english"))

    real_headlines = news_data[news_data["label"] == 0]["content"]
    fake_headlines = news_data[news_data["label"] == 1]["content"]

    def get_word_frequencies(headlines):
        all_words = []
        for headline in headlines:
            words = [
                word
                for word in headline.split()
                if word.isalpha() and word not in stop_words and len(word) > 1
            ]
            all_words.extend(words)
        return Counter(all_words)

    real_word_counts = get_word_frequencies(real_headlines)
    fake_word_counts = get_word_frequencies(fake_headlines)

    plt.figure(figsize=(14, 6))

    # Plot most common words for Real News
    plt.subplot(1, 2, 1)
    real_top_words = real_word_counts.most_common(top_n)
    plt.barh(
        [word for word, count in real_top_words],
        [count for word, count in real_top_words],
        color="skyblue",
    )
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Most Common Words in Real News Headlines")
    plt.xlabel("Frequency")
    plt.ylabel("Word")

    # Plot most common words for Fake News
    plt.subplot(1, 2, 2)
    fake_top_words = fake_word_counts.most_common(top_n)
    plt.barh(
        [word for word, count in fake_top_words],
        [count for word, count in fake_top_words],
        color="lightcoral",
    )
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Most Common Words in Fake News Headlines")
    plt.xlabel("Frequency")
    plt.ylabel("Word")

    plt.tight_layout()
    plot_file = os.path.join(save_path, "word_frequencies.png")
    plt.savefig(plot_file)
    print(f"Saved word frequencies plot to {plot_file}")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred_proba: np.ndarray, save_path: str, threshold: float = 0.5
):
    """
    Plot the confusion matrix and print the classification report.
    Save the plot to the specified path.
    """
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
    plot_file = os.path.join(save_path, "confusion_matrix.png")
    plt.savefig(plot_file)
    print(f"Saved confusion matrix plot to {plot_file}")
    plt.close()

    print("\nClassification Report:")
    print(
        classification_report(
            y_true, y_pred_classes, target_names=["Real News", "Fake News"]
        )
    )


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, save_path: str):
    """
    Plot the Receiver Operating Characteristic (ROC) curve and calculate AUC.
    Save the plot to the specified path.
    """
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
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plot_file = os.path.join(save_path, "roc_curve.png")
    plt.savefig(plot_file)
    print(f"Saved ROC curve plot to {plot_file}")
    plt.close()
