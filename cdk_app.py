import os
import aws_cdk as cdk

from fake_news_stack import FakeNewsClassifierStack
from config import config

app = cdk.App()

s3_bucket_name = config["s3_bucket_name"]
s3_model_key = config["s3_model_key"]
s3_tokenizer_key = config["s3_tokenizer_key"]
max_sequence_length = config["max_sequence_length"]
prediction_threshold = config["prediction_threshold"]

FakeNewsClassifierStack(
    app,
    "FakeNewsClassifierStack",
    s3_bucket_name=s3_bucket_name,
    s3_model_key=s3_model_key,
    s3_tokenizer_key=s3_tokenizer_key,
    max_sequence_length=max_sequence_length,
    prediction_threshold=prediction_threshold,
    lambda_memory_mb=config["lambda_memory_mb"],
    lambda_timeout_seconds=config["lambda_timeout_seconds"],
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
    ),
)

app.synth()
