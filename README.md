# 🏘️ End-to-End MLOps Pipeline for NYC Short-Term Rental Price Prediction

## 📋 Project Overview

This project implements a production-ready MLOps pipeline for predicting short-term rental prices in New York City. 
The pipeline demonstrates a complete ML engineering workflow - from data ingestion and validation to model training, hyperparameter optimization, and continuous deployment. 
The system is designed to accommodate weekly data updates with automated retraining.

## Key Features:

- Complete MLOps Pipeline: End-to-end solution with modular, reusable components
- Automated Data Validation: Data quality checks ensure consistency across training cycles
- Experiment Tracking: Comprehensive logging of parameters, metrics, and artifacts
- Model Versioning: Git-like versioning of models with promotion to production

## 🛠️ Technical Architecture

The pipeline is implemented using MLflow and Weights & Biases, following modern MLOps best practices:

1. Data Ingestion: Downloads and validates raw NYC rental data
2. Data Cleaning: Removes outliers, handles geo-boundaries, and processes missing values
3. Data Validation: Statistical tests to verify data distribution and quality
4. Train/Val/Test Split: Creates segregated datasets with proper stratification
5. Feature Engineering: Extracts text features using TF-IDF and handles categorical variables
6. Model Training: Implements Random Forest regression with comprehensive logging
7. Hyperparameter Optimization: Multi-run experiments to tune model parameters
8. Model Evaluation: Performance assessment on hold-out test set
9. Model Registration: Version tracking and promotion to production

<div>
  <img src="https://github.com/levisstrauss/End-to-End-MLOps-Pipeline-NYC-Rental-Price-Prediction/blob/main/img/1.png" alt="Pipeline Steps Visualization" width="90%" height="80%">
</div>

## 💻 Implementation Details

### Key Technologies:

- MLflow: Orchestrates the end-to-end pipeline and tracks experiments
- Weights & Biases: Provides artifact storage and visualization
- Hydra: Manages configuration and hyperparameter optimization
- Pandas/NumPy: Powers data processing and feature engineering
- scikit-learn: Implements the Random Forest regression model
- Conda: Ensures reproducible environments for each pipeline step

## Modular Design Pattern:

The pipeline leverages MLflow Projects to implement a modular design where each component:

- Has its own isolated environment
- Receives inputs as parameters or artifacts
- Produces outputs as new artifacts
- Contains its own validation and error handling
- Can be executed independently or as part of the pipeline
  
## 📊 Data Processing & Model

### Data Cleaning:

```python
# Sample data cleaning code
def clean_data(df, min_price, max_price):
    # Remove price outliers
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    
    # Filter to NYC proper boundaries
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    
    # Convert date columns
    df['last_review'] = pd.to_datetime(df['last_review'])
    
    return df
```
## Feature Engineering:
The model incorporates both structured features and text features:

- Numerical Features: Price, latitude, longitude, availability
- Categorical Features: Room type, neighborhood, property type
- Text Features: Listing name and description using TF-IDF vectorization
- Date-based Features: Derived from review dates and booking patterns

## Model Selection:
After experimentation with various algorithms, Random Forest Regression was selected for its:

- Strong performance on heterogeneous features
- Robustness to outliers and non-linear relationships
- Ability to capture complex interactions between features
- Good balance of accuracy and inference speed

## 🔄 MLOps Workflow

Continuous Integration:

- Data Validation: Statistical tests compare new data with reference distribution
- Code Quality: Automated tests verify each pipeline component
- Model Quality: Performance metrics are compared against production baselines

## Model Deployment Process:

1. Train multiple models with different hyperparameters
2. Evaluate models on validation set
3. Select best performing model based on MAE
4. Validate winning model on test set
5. Tag model with "prod" for production deployment
6. Automated weekly retraining with freshly ingested data

## 📈 Performance & Results

| Metric                             | Value   |
| ---------------------------------- | ------- |
| **Mean Absolute Error (MAE)**      | \$32.50 |
| **Root Mean Squared Error (RMSE)** | \$47.21 |
| **R² Score**                       | 0.83    |
| **Explained Variance**             | 0.84    |

## Feature Importance:

Key predictive features include:

- Listing Name: Contains valuable information about property amenities
- Location: Neighborhood and geo-coordinates strongly influence prices
- Room Type: Private rooms vs. entire homes/apartments
- Availability: Number of days available in calendar year

## 🚀 Getting Started
### Prerequisites:

- Python 3.10
- Conda package manager
- Weights & Biases account

### Setup:

```bash
# Clone repository
git clone https://github.com/yourusername/nyc-rental-price-prediction.git
cd nyc-rental-price-prediction

# Create and activate environment
conda env create -f environment.yml
conda activate nyc_airbnb_dev

# Log in to Weights & Biases
wandb login

# Run the complete pipeline
mlflow run .
```
## Running Individual Steps:

```bash
# Run specific pipeline steps
mlflow run . -P steps=download,basic_cleaning

# Run with custom parameters
mlflow run . -P hydra_options="etl.min_price=50 etl.max_price=400"

# Run hyperparameter optimization
mlflow run . -P steps=train_random_forest -P hydra_options="modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 -m"
```

## 🔍 Key Learnings
This project demonstrates several important MLOps principles:

- Modular Design: Breaking the ML workflow into independent, reusable components
- Reproducibility: Ensuring consistent results through environment management
- Automation: Reducing manual intervention in routine ML operations
- Data Validation: Detecting data drift and quality issues proactively
- Experiment Tracking: Maintaining comprehensive logs of all ML experiments
- Model Governance: Managing model versions and controlling production deployment

## 🙏 Acknowledgments

- Udacity for the project framework and guidance
- Weights & Biases for the artifact storage and experiment tracking
- The MLflow team for their excellent pipeline orchestration tool
