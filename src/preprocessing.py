import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import config
from src.custom_transformers import TextLowercaser


def remove_unneeded_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a 'content' column from 'title' and drop other unneeded features.
    Assumes 'content' is already lowercased (handled by TextLowercaser in main).
    """
    # The 'content' column should have been created and lowercased in main.py before this function is called.
    # So, we just need to drop the original columns.
    print("Removing unneeded columns: 'subject', 'date', 'title', 'text'...")
    return data.drop(columns=["subject", "date", "title", "text"], errors="ignore")


def tokenize_and_pad_text(data: pd.DataFrame):
    """
    Tokenize the 'content' column and pad sequences.
    """
    print("\nTokenizing data...")
    tokenizer = Tokenizer(num_words=config["tokenizer_num_words"], oov_token="<unk>")
    tokenizer.fit_on_texts(data["content"])
    word_index = tokenizer.word_index
    print(f"Found {len(word_index)} unique tokens in the dataset.")

    sequences = tokenizer.texts_to_sequences(data["content"])
    maxlen = max([len(seq) for seq in sequences])
    print(f"Max sequence length: {maxlen}")

    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding="post")
    print("Text tokenization and padding complete.")
    return padded_sequences, tokenizer, maxlen
