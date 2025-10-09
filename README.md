# 🏥 SmartPremium: Predicting Insurance Costs with Machine Learning

![Python](https://img.shields.io/badge/Python-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-red)
![MLflow](https://img.shields.io/badge/MLflow-green)
![XGBoost](https://img.shields.io/badge/XGBoost-brightgreen)
![Random%20Forest](https://img.shields.io/badge/Random%20Forest-blueviolet)


A comprehensive machine learning project that predicts insurance premiums based on customer characteristics and policy details. This end-to-end solution includes data analysis, model training, MLflow experiment tracking, and a Streamlit web application for real-time predictions.

## 🎯 Overview

Insurance companies use various factors such as **age**, **income**, **health status**, and **claim history** to estimate premiums for customers. This project builds a **machine learning model** that accurately predicts **insurance premiums** based on customer characteristics and policy details.

### Business Use Cases

- **🏢 Insurance Companies**: Optimize premium pricing based on risk factors
- **💳 Financial Institutions**: Assess risk for loan approvals tied to insurance policies  
- **🏥 Healthcare Providers**: Estimate future healthcare costs for patients
- **🤖 Customer Service Optimization**: Provide real-time insurance quotes based on data-driven predictions

## ✨ Features

### 🔧 Core Capabilities

- **📊 Comprehensive EDA** with interactive visualizations
- **🔄 Automated Data Preprocessing** with missing value handling
- **🎯 Advanced Feature Engineering** with domain-specific features
- **🤖 Multiple ML Algorithms** (Random Forest, XGBoost) with comparison
- **📈 MLflow Experiment Tracking** for model management
- **🌐 Streamlit Web Application** for real-time predictions
- **📁 Model Persistence** with artifact storage

### 📊 Data Analysis Features

- Target variable distribution analysis
- Correlation matrix and feature relationships
- Numerical and categorical feature distributions
- Feature importance analysis
- Model performance visualization

### 🌐 Web Application Features

- Real-time premium predictions
- User-friendly input interface
- Feature importance visualization
- Risk factor analysis
- Responsive design with custom styling

## 📁 Project Structure

```text
mini_project-4/
├─ data/
│  ├─ train.csv                    
│  ├─ test.csv                     
│  └─ sample_submission.csv     
│
├─ main.py                         # Training pipeline with MLflow
├─ streamlit_app.py                # Streamlit web application
│
├─ outputs/
│  ├─ final_submission.csv         # Predictions
│  ├─ model_artifacts.pkl          # Trained model & preprocessors
│  ├─ target_distribution.png      # EDA visualizations
│  ├─ numerical_distributions.png
│  ├─ categorical_distributions.png
│  ├─ correlation_matrix.png
│  └─ feature_importance.png
│
├─ requirements.txt                # Project dependencies
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/mini_project-4.git
cd mini_project-4
```

### Step 2: Install the Dependencies
```bash
pip install -r requirements.txt
```

## Complete Workflow
### 1. Data Preparation
### Ensure data files are in project root:
```bash
# - train.csv
# - test.csv  
# - sample_submission.csv
```
### 2. Run Training Pipeline
```bash
python main.py
```
### 3. Monitor Experiments
```bash
mlflow ui
```
### 4. Deploy Web App
```bash
streamlit run streamlit_app.py
```

## Model Training Details
### The pipeline includes:
```bash 
1. Data Sampling: 10-20% of data for faster iteration

2. Feature Engineering: Interaction terms, demographic groups, risk flags

3. Model Comparison: Random Forest vs XGBoost with hyperparameters

4. Evaluation: RMSE, MAE, R² metrics

5. Artifact Saving: Models, scalers, encoders persisted
```

## Evaluation Metrics
```bash
RMSE (Root Mean Squared Error): √(Σ(yᵢ - ŷᵢ)²/n)

MAE (Mean Absolute Error): Σ|yᵢ - ŷᵢ|/n

R² Score: Proportion of variance explained
```
## Feature Engineering
```bash
Interaction Features: Age×Health, Income×Credit

Demographic Groups: Age categories, Income brackets

Risk Flags: Young driver, Senior citizen, Poor health

Encoding: Label encoding for categorical variables

```
