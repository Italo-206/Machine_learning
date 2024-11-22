# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing  # Using California Housing dataset

# 1. Load and prepare the dataset
def load_and_prepare_data():
    print("Loading the dataset...")
    try:
        # Load the California Housing dataset
        california = fetch_california_housing(as_frame=True)
        data = california.data
        data['PRICE'] = california.target  # Add the target variable (price)
        print("Dataset loaded successfully.")
        
        # Check for missing values
        if data.isnull().sum().any():
            print("Warning: The dataset contains missing values. Handle them before proceeding.")
            return None
        
        print(f"The dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except Exception as e:
        print(f"Error loading the dataset: {e}")
        return None

# 2. Perform exploratory data analysis (EDA)
def exploratory_data_analysis(data):
    print("\nStarting exploratory data analysis...")
    try:
        # Display the first few rows of the dataset
        print("First few rows of the dataset:")
        print(data.head())

        # Display basic information (data types, missing values)
        print("\nDataset information:")
        print(data.info())

        # Display descriptive statistics
        print("\nDescriptive statistics:")
        print(data.describe())

        # Plot a heatmap of the correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        plt.title("Matriz de Correlação")  # Translated to Portuguese
        plt.show()
        print("Exploratory data analysis completed successfully.")
    except Exception as e:
        print(f"Error during exploratory data analysis: {e}")

# 3. Split data into training and testing sets
def split_data(data):
    print("\nSplitting the dataset into training and testing sets...")
    try:
        # Separate features (X) and target (y)
        X = data.drop('PRICE', axis=1)  # Features
        y = data['PRICE']  # Target variable

        # Split into 80% training and 20% testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data successfully split: 80% training, 20% testing.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error splitting the dataset: {e}")
        return None, None, None, None

# 4. Train a Linear Regression model
def train_model(X_train, y_train):
    print("\nTraining the Linear Regression model...")
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)  # Train the model
        print("Model training completed.")
        return model
    except Exception as e:
        print(f"Error training the model: {e}")
        return None

# 5. Evaluate the model on test data
def evaluate_model(model, X_test, y_test):
    print("\nEvaluating the model...")
    try:
        y_pred = model.predict(X_test)  # Make predictions on the test set

        # Compute evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\nModel Evaluation:")
        print(f"Erro Absoluto Médio (MAE): {mae:.2f}")  # In Portuguese
        print(f"Erro Quadrático Médio (MSE): {mse:.2f}")  # In Portuguese
        print(f"Coeficiente de Determinação (R²): {r2:.2f}")  # In Portuguese

        # Plot a comparison of actual vs predicted values
        plt.scatter(y_test, y_pred)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
        plt.xlabel('Valores Reais')  # Translated to Portuguese
        plt.ylabel('Valores Previstos')  # Translated to Portuguese
        plt.title('Comparação: Valores Reais vs Previstos')  # Translated to Portuguese
        plt.show()
        print("Model evaluation completed successfully.")
    except Exception as e:
        print(f"Error during model evaluation: {e}")

# Main function to run the full pipeline
def main():
    # Load and prepare the dataset
    data = load_and_prepare_data()
    if data is None:
        print("Error: Dataset could not be loaded. Exiting the program.")
        return

    # Perform exploratory data analysis
    exploratory_data_analysis(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(data)
    if X_train is None or X_test is None:
        print("Error: Could not split the dataset. Exiting the program.")
        return

    # Train the model
    model = train_model(X_train, y_train)
    if model is None:
        print("Error: Model training failed. Exiting the program.")
        return

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

# Run the main function
if __name__ == "__main__":
    main()
