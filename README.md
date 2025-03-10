# Atelier 1: Deep Learning for Regression and Classification

**Developed by:** Amine Izougaghen  
**Supervised by:** Prof. EL ACHAAK Lotfi

![Project Badge](https://img.shields.io/badge/Deep%20Learning-PyTorch-blue)

## Table of Contents

1. [Objective](#objective)
2. [Part 1: Stock Market Regression Analysis](#part-1-stock-market-regression-analysis)
   - [Objective](#objective-1)
   - [Dataset](#dataset-1)
   - [Implementation](#implementation-1)
     - [Data Preprocessing](#data-preprocessing-1)
     - [Model Training](#model-training-1)
     - [Evaluation & Visualization](#evaluation--visualization-1)
   - [Results & Discussion](#results--discussion-1)
   - [Future Enhancements](#future-enhancements-1)
3. [Part 2: Predictive Maintenance Classification](#part-2-predictive-maintenance-classification)
   - [Objective](#objective-2)
   - [Dataset](#dataset-2)
   - [Implementation](#implementation-2)
     - [Data Preprocessing](#data-preprocessing-2)
     - [Model Training](#model-training-2)
     - [Evaluation & Visualization](#evaluation--visualization-2)
   - [Results & Discussion](#results--discussion-2)
   - [Future Enhancements](#future-enhancements-2)
4. [How to Run the Project](#how-to-run-the-project)
   - [Prerequisites](#prerequisites)
   - [Running the Model](#running-the-model)

---

## Objective

The main purpose of this lab is to get familiar with the PyTorch library to perform classification and regression tasks by establishing DNN/MLP architectures.

---

## Part 1: Stock Market Regression Analysis

### Objective

Develop a deep learning model to predict stock closing prices based on historical market data, aiding in investment strategy and risk management.

### Dataset

- **Source**: Kaggle - NYSE Dataset
- **Features**: Date, Symbol, Open, Close (Target), Low, High, Volume
- **Data Considerations**: Time-series data requiring temporal analysis.

### Implementation

#### Data Preprocessing

1. **Loading & Cleaning**: Handle missing values via interpolation or imputation.
2. **Feature Engineering**:
   - Moving Averages (5-day, 20-day)
   - Relative Strength Index (RSI)
   - Moving Average Convergence Divergence (MACD)
3. **Scaling**: Normalize data using `StandardScaler` or `MinMaxScaler`.

#### Model Training

1. **Data Splitting**: 70% Training, 15% Validation, 15% Testing (time-based split).
2. **Model Architectures**:
   - Multi-Layer Perceptron (MLP)
   - Long Short-Term Memory (LSTM) or Gated Recurrent Units (GRUs) for sequential dependencies.
   - Convolutional Neural Networks (CNNs) for pattern extraction.
3. **Optimization**:
   - **Loss Function**: Mean Squared Error (MSE)
   - **Optimizer**: Adam or RMSprop
   - **Regularization**: Early stopping to prevent overfitting.
   - **Hyperparameter Tuning**: Grid search or random search.

#### Evaluation & Visualization

1. **Metrics**: MSE, RMSE, R², MAE
2. **Visualization**: Compare actual vs. predicted prices, analyze residuals.

#### Results & Discussion

1. **Performance Analysis**: Report key metrics and discuss model accuracy.
2. **Strengths & Weaknesses**: Identify potential improvements and limitations.
3. **Prediction Analysis**: Examine reasons for deviations in stock price predictions.

#### Future Enhancements

1. **Feature Engineering**: Add macroeconomic indicators and sentiment analysis.
2. **Advanced Models**: Explore Transformers and ensemble methods.
3. **Optimization**: Use Bayesian optimization for hyperparameter tuning.
4. **Backtesting & Risk Management**: Simulate real-world trading scenarios and incorporate risk strategies.

---

## Part 2: Predictive Maintenance Classification

### Objective

Develop a deep learning model that classifies machine conditions to predict potential failures and improve maintenance scheduling.

### Dataset

- **Source**: Kaggle - Predictive Maintenance
- **Features**: Sensor readings, machine parameters, and failure labels.
- **Data Considerations**: Identify key features for accurate classification.

### Implementation

#### Data Preprocessing

1. **Data Cleaning**: Handle missing values using imputation techniques.
2. **Feature Selection**:
   - Correlation analysis to remove redundant features.
   - Feature importance using tree-based models.
   - Statistical tests (e.g., chi-squared test, ANOVA).
3. **Scaling**: Normalize input features using `StandardScaler` or `MinMaxScaler`.
4. **Data Splitting**: Training, validation, and testing sets.

#### Model Training

1. **Model Architectures**:
   - Multi-Layer Perceptron (MLP)
   - Convolutional Neural Networks (CNNs)
   - Recurrent Neural Networks (RNNs) for sequential dependencies.
2. **Optimization**:
   - **Loss Function**: Categorical Cross-Entropy
   - **Optimizer**: Adam
   - **Regularization**: Dropout layers and early stopping.
   - **Hyperparameter Tuning**: Grid search or random search.

#### Evaluation & Visualization

1. **Metrics**: Accuracy, Precision, Recall, F1-score
2. **Visualization**: Confusion matrix, AUC-ROC curve.

#### Results & Discussion

1. **Performance Analysis**: Summarize accuracy and key metrics.
2. **Error Analysis**: Examine misclassified cases.
3. **Potential Improvements**: Data augmentation, advanced architectures (ResNet, Transformers), and hyperparameter tuning.

#### Future Enhancements

1. **Ensemble Learning**: Combine multiple models for better accuracy.
2. **Explainable AI (XAI)**: Use SHAP/LIME to interpret model decisions.
3. **Online Learning**: Update models with real-time data for continuous improvement.

---

## **How to Run the Project**

### **Prerequisites**
Ensure you have the following dependencies installed:
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn imbalanced-learn

```

### **Running the Model**
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <project_folder>
   ```
2. Load the dataset and preprocess the data.
3. Train the model using the defined architectures.
4. Evaluate the model and visualize predictions.

---

This project demonstrates deep learning’s effectiveness in financial forecasting and provides a framework for future improvements in predictive modeling for stock market analysis.

