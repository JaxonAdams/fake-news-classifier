import os
import json

import boto3
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

from custom_transformers import TextLowercaser


# Global variables to load model and tokenizer once per cold start
MODEL = None
TOKENIZER = None
MAXLEN = None
TEXT_LOWERCASE_TRANSFORMER = TextLowercaser()
PREDICTION_THRESHOLD = 0.5

s3_client = boto3.client("s3")

LOCAL_MODEL_PATH = "/tmp/LSTM.keras"
LOCAL_TOKENIZER_PATH = "/tmp/tokenizer.json"


def load_assets_from_s3():
    """
    Download the model and tokenizer from S3 to Lambda's /tmp directory.
    """
    global MODEL, TOKENIZER, MAXLEN, PREDICTION_THRESHOLD

    s3_bucket_name = os.environ.get("S3_BUCKET_NAME")
    s3_model_key = os.environ.get("S3_MODEL_KEY")
    s3_tokenizer_key = os.environ.get("S3_TOKENIZER_KEY")
    max_sequence_length_str = os.environ.get("MAX_SEQUENCE_LENGTH")
    prediction_threshold_str = os.environ.get("PREDICTION_THRESHOLD", 0.5)

    if not all(
        [s3_bucket_name, s3_model_key, s3_tokenizer_key, max_sequence_length_str]
    ):
        raise ValueError(
            json.dumps(
                {
                    "message": "Missing one or more required environment variables",
                    "metadata": {
                        "missing": [
                            "S3_BUCKET_NAME",
                            "S3_MODEL_KEY",
                            "S3_TOKENIZER_KEY",
                            "MAX_SEQUENCE_LENGTH",
                        ],
                    },
                },
                indent=2,
            )
        )

    MAXLEN = int(max_sequence_length_str)
    PREDICTION_THRESHOLD = float(prediction_threshold_str)

    if not os.path.exists(LOCAL_MODEL_PATH) or MODEL is None:
        print(
            f"Downloading model from s3://{s3_bucket_name}/{s3_model_key} to {LOCAL_MODEL_PATH}..."
        )
        s3_client.download_file(s3_bucket_name, s3_model_key, LOCAL_MODEL_PATH)
        MODEL = tf.keras.models.load_model(LOCAL_MODEL_PATH)
        print("Model loaded.")
    else:
        print("Model already loaded or exists in /tmp/, skipping download/reload.")

    if not os.path.exists(LOCAL_TOKENIZER_PATH) or TOKENIZER is None:
        print(
            f"Downloading tokenizer from s3://{s3_bucket_name}/{s3_tokenizer_key} to {LOCAL_TOKENIZER_PATH}..."
        )
        s3_client.download_file(s3_bucket_name, s3_tokenizer_key, LOCAL_TOKENIZER_PATH)
        with open(LOCAL_TOKENIZER_PATH, "r", encoding="utf-8") as f:
            tokenizer_json = json.load(f)
        TOKENIZER = tokenizer_from_json(tokenizer_json)
        print("Tokenizer loaded.")
    else:
        print("Tokenizer already loaded or exists in /tmp/, skipping download/reload.")


def lambda_handler(event, context):
    """
    Lambda function handler for fake news prediction.
    Expects a JSON body with a 'headline' field.
    """
    try:
        if MODEL is None or TOKENIZER is None or MAXLEN is None:
            load_assets_from_s3()

        body = json.loads(event["body"])
        news_headline = body.get("headline")

        if not news_headline:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": 'Missing "headline" in request body.'}),
            }

        # Preprocess the input headline
        processed_headline = TEXT_LOWERCASE_TRANSFORMER.transform([news_headline])
        sequence = TOKENIZER.texts_to_sequences(processed_headline)

        if not sequence or not sequence[0]:
            print(
                f"Warning: Input headline '{news_headline}' resulted in an empty sequence after tokenization."
            )
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(
                    {
                        "headline": news_headline,
                        "is_fake_news": False,
                        "confidence": 0.5,
                        "message": "Cannot classify: Headline contains no recognized words. Defaulting to not fake.",
                    }
                ),
            }

        padded_sequence = pad_sequences(sequence, maxlen=MAXLEN, padding="post")

        # Make a prediction
        prediction = MODEL.predict(padded_sequence)[0][0]
        is_fake = bool(prediction > PREDICTION_THRESHOLD)
        confidence = float(prediction)

        response_body = {
            "headline": news_headline,
            "is_fake_news": is_fake,
            "confidence": confidence,
            "message": "LIKELY FAKE!" if is_fake else "Not likely to be fake.",
        }

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
            },
            "body": json.dumps(response_body),
        }
    except Exception as e:
        print(f"Error processing request: {e}")
        import traceback

        traceback.print_exc()
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error", "details": str(e)}),
        }


load_assets_from_s3()
