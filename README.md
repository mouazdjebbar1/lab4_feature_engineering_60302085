# lab4_feature_engineering_60302085
# Azure ML Assignment 2 – Model Training & Automation

## 1. Feature Engineering (Lab 4)

We used the pipeline from Lab 4 to generate features from the Amazon Electronics dataset.

The pipeline includes:
- Text normalization  
- Length features  
- Sentiment features  
- TF-IDF features  
- SBERT embeddings  
- Merging all features into final datasets  

At the end, we obtained:
- Train dataset  
- Validation dataset  
- Test dataset  
- Deployment dataset (10%)  

All datasets were saved as `.parquet` files and registered in Azure ML.

---

## 2. Data Registration

Each dataset was registered as a data asset in Azure ML:

- `amazon_review_merged_features_train`  
- `amazon_review_merged_features_val`  
- `amazon_review_merged_features_test`  
- `amazon_review_merged_features_deploy`  

These datasets are used as inputs for training and deployment.

---

## 3. Model Training (`train.py`)

We created a training script that performs the following steps:

- Loads train, validation, and test datasets  
- Converts `overall` rating into a binary label:  
  - `1` → rating ≥ 4  
  - `0` → rating < 4  
- Selects only useful numeric features (drops text and identifiers)  
- Trains a Logistic Regression model  
- Evaluates performance on:
  - Train set  
  - Validation set  
  - Test set  

### Metrics logged using MLflow:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- AUC  

The trained model is saved as:

    model.pkl

---

## 4. Training Job (Azure ML)

We submitted the training job using:

    az ml job create --file jobs/train_job.yml

This runs the training script on Azure ML compute and logs results to MLflow.

---

## 5. Hyperparameter Support

We modified `train.py` to accept hyperparameters:

- `C` (regularization strength)  
- `max_iter` (maximum iterations)  

These are passed via command line and logged with MLflow.

---

## 6. Sweep Job (Hyperparameter Tuning)

We created a sweep job to automatically find the best hyperparameters.

### Search Space:
- `C`: continuous range (0.1 → 3.0)  
- `max_iter`: [500, 1000, 1500]  

### Objective:
- Maximize `val_accuracy`  

### Command:

    az ml job create --file jobs/sweep_job.yml

The sweep runs multiple trials and selects the best configuration.

---

## 7. Best Hyperparameters (From Sweep)

The best performing run gave:

- `C = 0.9121`  
- `max_iter = 1500`  

These values were used for the final model.

---

## 8. Final Model Training

We retrained the model using the best hyperparameters.

This ensures:
- Better generalization  
- Improved validation performance  

---

## 9. Model Registration

The final trained model was registered in Azure ML:

    amazon-review-sentiment-model

This model is used for deployment.

---

## 10. Model Deployment (Online Endpoint)

We deployed the model as a real-time endpoint.

### Steps:

Create endpoint:

    az ml online-endpoint create --name amazon-review-endpoint --auth-mode key

Create deployment:

    az ml online-deployment create --file jobs/deployment.yml --all-traffic

Fix deployment issues:
- Corrected model path (`model_output/model.pkl`)  
- Fixed `score.py` loading logic  

Assign traffic:

    az ml online-endpoint update --name amazon-review-endpoint --traffic blue=100

---

## 11. Scoring Script (`score.py`)

The scoring script:

- Loads the model during initialization  
- Applies the same feature selection logic as training  
- Accepts JSON input  
- Returns predictions  

---

## 12. Endpoint Testing

We tested the deployed model using `invoke_endpoint.py`.

### Steps:
- Loaded deployment dataset (`data.parquet`)  
- Selected a small sample (5 rows)  
- Sent request to the endpoint  
- Received predictions  

### Result:
- Status Code: **200**  
- Model successfully returned predictions  

---

## 13. Workflow Summary

1. Feature engineering pipeline (Lab 4)  
2. Dataset registration in Azure ML  
3. Training script implementation  
4. Training job execution  
5. Hyperparameter tuning (sweep job)  
6. Best model selection  
7. Final model retraining  
8. Model registration  
9. Endpoint deployment  
10. Endpoint testing with real data  

---

## 14. Notes

- MLflow is used for tracking metrics and parameters  
- Validation set is used for tuning  
- Test set is used for final evaluation only  
- Deployment dataset is used to test the endpoint  
- Only numeric features are used for training and inference  
- Endpoint requires JSON input (not Azure dataset paths)  

---

## 15. Conclusion

We successfully built a complete machine learning pipeline using Azure ML, including:

- Data processing  
- Model training  
- Hyperparameter tuning  
- Model registration  
- Real-time deployment  
- Endpoint testing  

The system is fully functional and can predict review sentiment from input data.
