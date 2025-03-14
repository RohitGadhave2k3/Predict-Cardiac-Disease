# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_data():
    """
    Load and prepare the heart disease dataset
    Returns:
        DataFrame: The loaded dataset
    """
    # Load the dataset from CSV file
    df = pd.read_csv('heart_disease_data.csv')
    
    # Data dictionary for reference
    feature_descriptions = {
        'age': 'Age in years',
        'sex': 'Sex (1 = male, 0 = female)',
        'cp': 'Chest pain type (0-3)',
        'trestbps': 'Resting blood pressure in mm Hg',
        'chol': 'Serum cholesterol in mg/dl',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
        'restecg': 'Resting ECG results (0-2)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes, 0 = no)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of the peak exercise ST segment (0-2)',
        'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
        'thal': 'Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)',
        'target': 'Heart disease diagnosis (1 = present, 0 = absent)'
    }
    
    print("Dataset Features Description:")
    for feature, description in feature_descriptions.items():
        print(f"{feature}: {description}")
    
    return df

def explore_data(df):
    """
    Perform exploratory data analysis
    Args:
        df (DataFrame): Input dataset
    """
    print("\nDataset Info:")
    print("-" * 50)
    print(df.info())
    
    print("\nBasic Statistics:")
    print("-" * 50)
    print(df.describe())
    
    print("\nMissing Values:")
    print("-" * 50)
    print(df.isnull().sum())
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # Distribution of target variable
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='target')
    plt.title('Distribution of Heart Disease Cases')
    plt.savefig('target_distribution.png')
    plt.close()

def preprocess_data(df):
    """
    Preprocess the data for modeling
    Args:
        df (DataFrame): Input dataset
    Returns:
        tuple: Preprocessed features and target variables, and the scaler object
    """
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple models
    Args:
        X_train, X_test, y_train, y_test: Training and testing data
    Returns:
        dict: Dictionary containing trained models and their performances
    """
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier()
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': report
        }
        
        print(f"\n{name} Results:")
        print("-" * 50)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
    
    return results

def predict_heart_disease(model, scaler, user_input):
    """
    Make predictions for user input
    Args:
        model: Trained model
        scaler: Fitted scaler
        user_input: Dictionary of user input values
    Returns:
        int: Prediction (0 or 1)
    """
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Scale the input
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    return prediction[0]

def main():
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Explore data
    print("\nPerforming exploratory data analysis...")
    explore_data(df)
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_scaled, y, scaler = preprocess_data(df)
    
    # Split data
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train and evaluate models
    print("\nTraining and evaluating models...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest performing model: {best_model[0]}")
    print(f"Accuracy: {best_model[1]['accuracy']:.4f}")
    
    # Example prediction
    print("\nExample prediction with best model:")
    example_input = {
        'age': 55,
        'sex': 1,
        'cp': 2,
        'trestbps': 130,
        'chol': 240,
        'fbs': 0,
        'restecg': 1,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 1.5,
        'slope': 2,
        'ca': 0,
        'thal': 2
    }
    
    prediction = predict_heart_disease(best_model[1]['model'], scaler, example_input)
    print(f"Prediction for example input: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")

if __name__ == "__main__":
    main() 