# Use the Amazon Linux 2 base image for Python 3.10 from ECR Public Gallery
FROM public.ecr.aws/lambda/python:3.10

# Set the working directory in the container
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy the requirements file and install dependencies
COPY server/requirements.txt .
RUN pip install -r requirements.txt --target .

# Copy the application code and custom transformer
COPY server/predict_function.py .
COPY custom_transformers.py .

CMD [ "predict_function.lambda_handler" ]
