import os

import aws_cdk as cdk
from constructs import Construct
from aws_cdk import (
    aws_lambda as lambda_,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_apigateway as apigw,
    Duration,
    RemovalPolicy,
)


class FakeNewsClassifierStack(cdk.Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        s3_bucket_name: str,
        s3_model_key: str,
        s3_tokenizer_key: str,
        max_sequence_length: int,
        prediction_threshold: float,
        lambda_memory_mb: int,
        lambda_timeout_seconds: int,
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # S3 bucket to store model assets
        model_assets_bucket = s3.Bucket(
            self,
            "ModelAssetsBucket",
            bucket_name=s3_bucket_name,
            removal_policy=cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )
        cdk.CfnOutput(
            self, "ModelAssetsBucketName", value=model_assets_bucket.bucket_name
        )

        # Lambda function using Docker image
        fake_news_lambda = lambda_.DockerImageFunction(
            self,
            "FakeNewsPredictorLambda",
            code=lambda_.DockerImageCode.from_image_asset(
                directory=os.path.join(
                    os.path.dirname(__file__),
                    "src",
                )
            ),
            memory_size=lambda_memory_mb,
            timeout=Duration.seconds(lambda_timeout_seconds),
            environment={
                "S3_BUCKET_NAME": model_assets_bucket.bucket_name,
                "S3_MODEL_KEY": s3_model_key,
                "S3_TOKENIZER_KEY": s3_tokenizer_key,
                "MAX_SEQUENCE_LENGTH": str(max_sequence_length),
                "PREDICTION_THRESHOLD": str(prediction_threshold),
            },
        )

        # Grant Lambda permissions to read from the S3 bucket
        model_assets_bucket.grant_read(fake_news_lambda)

        # API Gateway to expose the lambda function
        api = apigw.RestApi(
            self,
            "FakeNewsApi",
            rest_api_name="Fake News Predictor API",
            description="API for predicting fake news headlines.",
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
                allow_headers=[
                    "Content-Type",
                    "X-Amz-Date",
                    "Authorization",
                    "X-Api-Key",
                    "X-Amz-Security-Token",
                ],
            ),
        )

        # Add a POST method to the /predict path
        predict_resource = api.root.add_resource("predict")
        predict_resource.add_method(
            "POST",
            apigw.LambdaIntegration(
                fake_news_lambda,
                proxy=True,
                integration_responses=[
                    {
                        "statusCode": "200",
                        "responseParameters": {
                            "method.response.header.Access-Control-Allow-Origin": "'*'",
                        },
                    }
                ],
                request_templates={"application/json": '{ "statusCode": 200 }'},
            ),
            method_responses=[
                {
                    "statusCode": "200",
                    "responseParameters": {
                        "method.response.header.Access-Control-Allow-Origin": True
                    },
                }
            ],
        )

        cdk.CfnOutput(
            self,
            "ApiUrl",
            value=api.url_for_path("/predict"),
        )

        # S3 bucket for hosting the static frontend
        frontend_bucket = s3.Bucket(
            self,
            "FakeNewsClassifierFrontendBucket",
            website_index_document="index.html",
            website_error_document="index.html",
            public_read_access=True,
            block_public_access=s3.BlockPublicAccess(
                block_public_acls=False,
                ignore_public_acls=False,
                block_public_policy=False,
                restrict_public_buckets=False,
            ),
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            bucket_name="fake-news-classifier-jaxonadams-frontend",
        )

        s3deploy.BucketDeployment(
            self,
            "FakeNewsClassifierFrontendDeploy",
            sources=[s3deploy.Source.asset("frontend/public")],
            destination_bucket=frontend_bucket,
        )

        cdk.CfnOutput(
            self,
            "FrontendUrl",
            value=frontend_bucket.bucket_website_url,
            description="The URL of the frontend static website",
        )
