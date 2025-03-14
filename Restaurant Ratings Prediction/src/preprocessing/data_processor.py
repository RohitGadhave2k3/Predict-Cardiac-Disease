import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """
        Load restaurant data from CSV file
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        # Fill numeric columns with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Fill categorical columns with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
        
        return df
    
    def encode_categorical_features(self, df, categorical_columns):
        """
        Encode categorical features using LabelEncoder
        
        Args:
            df (pd.DataFrame): Input dataframe
            categorical_columns (list): List of categorical column names
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        df_encoded = df.copy()
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
            df_encoded[column] = self.label_encoders[column].fit_transform(df[column])
        return df_encoded
    
    def scale_features(self, df, feature_columns):
        """
        Scale numerical features using StandardScaler
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_columns (list): List of feature column names
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        df_scaled = df.copy()
        df_scaled[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        return df_scaled
    
    def prepare_data(self, df, target_column, feature_columns, categorical_columns, test_size=0.2):
        """
        Prepare data for model training
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target column
            feature_columns (list): List of feature column names
            categorical_columns (list): List of categorical column names
            test_size (float): Proportion of test set
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, categorical_columns)
        
        # Scale features
        df = self.scale_features(df, feature_columns)
        
        # Split features and target
        X = df[feature_columns]
        y = df[target_column]
        
        # Split into train and test sets
        return train_test_split(X, y, test_size=test_size, random_state=42) 