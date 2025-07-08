# Fake News Classifier
This project presents a comprehensive, end-to-end solution for classifying news headlines as "Real News" or "Fake News." It demonstrates proficiency in machine learning model development, cloud infrastructure deployment using Infrastructure as Code (IaC), serverless architecture, API design, and modern web frontend development.
![image](https://github.com/user-attachments/assets/5cd82575-c2d0-4ba3-b9d8-05d4b5d04095)
---
## Model Details & Performance
The core of this project is an LSTM (Long Short-Term Memory) neural network, a type of recurrent neural network particularly well-suited for sequence data like text.
### Dataset:
The model was trained and evaluated on a publicly-available [dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) of news headlines, separated as "real" and "fake".
### Preprocessing
Headlines undergo a series of preprocessing steps, including lowercasing, tokenization, and padding to a fixed sequence length to prepare them for the LSTM model. A custom `TextLowercaser` transformer was implemented for this purpose.
### Quantitative Performance (on Held-Out Test Set):
The model's performance was rigorously evaluated on a dedicated test set (8,980 samples) that was entirely unseen during training. The results demonstrate strong generalization capabilities:
| Metric         | Value    |
| -------------- | -------- |
| Test Accuracy  | 0.9679   |
| Test Loss      | 0.2444   |
### Classification Report:
```
              precision    recall  f1-score   support

  Real News       0.97      0.96      0.97      4330
  Fake News       0.96      0.97      0.97      4650

   accuracy                           0.97      8980
  macro avg       0.97      0.97      0.97      8980
weighted avg      0.97      0.97      0.97      8980
```
These metrics indicate that the model achieves **consistently high precision, recall, and F1-scores for both "Real News" and "Fake News" classes**, suggesting a robust ability to correctly identify and destingquish between them on data similar to its training distribution.
### Performance Visualizations:
#### ROC Curve / AUC
![roc_curve](https://github.com/user-attachments/assets/97c38bc8-33b4-4f7d-a026-1078da6ec97c)
#### Confusion Matrix
![confusion_matrix](https://github.com/user-attachments/assets/4a83822d-87d6-4608-b226-aec3e85feb5f)
#### Training Validation Performance
![training_validation_performance](https://github.com/user-attachments/assets/2f065428-86a4-4c88-948c-e18e7ec82e4e)
### Real-World Performance & Critical Analysis:
While the quantitative measures are excellent, manual testing with diverse, real-world headlines (e.g., from CNN, Fox News, and satirical sources like The Onion) revealed that the model's performance can sometimes differ from the reported test accuracy.
This observation is a valuable insight into the practical challenges of deploying ML models:
- **Data Distribution Shift**: The model's training data may not fully capture the stylistic nuances, evolving language, or specific characteristics of all real-world news content, particularly satirical news (like from The Onion), which is intentionally crafted to mimic legitimate news, differing significantly from common "fake news" patterns in the training dataset.
- **Definition of "Fake News"**: The model's definition of "fake news" is derived solely from its training data, which might not encompass the full spectrum of human interpretation (e.g., distinguishing satire from malicious misinformation).
This project demonstrates an understanding that high offline metrics do not always translate perfectly to real-world scenarios, emphasizing the importance of continuous monitoring, representative data collection, and iterative model improvement in MLOps.
Still, the metrics above demonstrate that the project's initial goal of creating a model capable of distinguishing between fake and real news was met, and its performance is admirable.

## Key Features
- **Machine Learning Model**: A deep learning LSTM model trained to classify news headlines.
- **Serverless Inference**: Model inference served via AWS Lambda, packaged as a Docker image for dependency management.
- **Scalable API**: Exposed through AWS API Gateway, enabling low-latency predictions.
- **Web-Based UI**: A responsive web frontend built with ClojureScript and Reagent, deployed on AWS S3.
- **Infrastructure as Code (IaC)**: All AWS infrastructure defined and deployed using AWS Cloud Development Kit (CDK) in Python, ensuring reproducibility and easy management.
- **Automated Asset Management**: Model and tokenizer assets stored in a private S3 bucket and securely accessed by the Lambda inference function.

## Architecture Overview
This solution leverages a serverless architecture on AWS, designed for scalability, cost-efficiency, and ease of deployment.
### Components:
- **AWS S3 (Frontend Bucket)**: Hosts the static ClojureScript web application. Configured for public read access as a static website.
- **AWS S3 (Model Assets Bucket)**: A private S3 bucket securely stores the trained Keras model (LSTM.keras) and tokenizer (tokenizer.json). The Lambda function is granted specific read permissions to download these assets during cold starts.
- **AWS API Gateway**: Acts as the public-facing HTTP endpoint for the prediction service. It handles incoming requests, manages CORS, and routes them to the Lambda function.
- **AWS Lambda**: The core of the prediction service. The Python-based ML inference code is deployed as a Docker image, allowing for complex dependencies (TensorFlow, NumPy, scikit-learn).
- **AWS Cloud Development Kit (CDK)**: Python is used to define and provision all AWS resources (Lambda, API Gateway, S3 buckets, IAM roles) as code, facilitating version control, automated deployments, and environment consistency.

## Technical Stack
### Backend & Infrastructure:
- **Language**: Python 3.10
- **Machine Learning Framework**: Tensorflow 2.x, Keras
- **Data Manipulation**: Pandas, NumPy, scikit-learn
- **Cloud Platform**: AWS
- **Infrastructure as Code (IaC)**: AWS Cloud Development Kit (CDK) v2
- **Containerization**: Docker
- **Asset Management**: AWS S3
### Frontend
- **Language**: ClojureScript
- **UI Framework**: Reagent (a minimalist React wrapper for ClojureScript)
- **Build Tool**: Shadow-CLJS (for ClojureScript compilation)
- **HTTP Client**: `lambdaisland/fetch` (modern, Promise-based API)
- **Styling**: Tailwind CSS (via CDN for simplicity, with awareness of production best practices)

## Deployment & MLOps Considerations
This project was built with deployment and operational considerations in mind:
- **Serverless Efficiency**: AWS Lambda and API Gateway provide a highly scalable and cost-effective solution, only consuming resources when requests are made.
- **Docker for Dependencies**: Packaging the Lambda function as a Docker image simplifies dependency management for complex ML libraries like TensorFlow, ensuring a consistent runtime environment.
- **Asset Management**: Storing the model and tokenizer in S3 and loading them during Lambda cold starts optimizes the deployment package size and allows for easy model updates without redeploying the entire Lambda code.
- **Cold Start Optimization**: Initial challenges with Lambda cold start timeouts due to large model loading were successfully addressed by optimizing Lambda memory allocation, demonstrating practical debugging and performance tuning skills.

## Future Enhancements
- **Advanced Model Architectures**: I'd like to experiment with Transformer-based models for potentially higher accuracy and better generalization.
- **Continuous Learning Pipeline**: I'd love to implement a feedback mechanism in the frontend to collect user corrections, enabling continuous model retraining and improvement.
- **Model Interpretability**: I could integrate tools to explain model predictions, providing transparency into why a headline is classified as real or fake.
- **Custom Domain & CDN**: Soon I will configure AWS Route 53 and CloudFront for a custom domain, SSL, and global content delivery network for the frontend.
- **CI/CD Pipeline**: I want to automate the entire deployment process (backend and frontend) using AWS CodePipeline or GitHub Actions.

---
Many thanks to `clmentbisaillon` for providing this dataset! (retrieved from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset))
