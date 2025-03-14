import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
import os

class ModelTrainer:
    def __init__(self, model_type='random_forest'):
        """
        Initialize the model trainer
        
        Args:
            model_type (str): Type of model to train ('random_forest', 'logistic', 'xgboost')
        """
        self.model_type = model_type
        self.model = self._create_model()
        
    def _create_model(self):
        """
        Create the specified model
        
        Returns:
            object: Initialized model
        """
        if self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'logistic':
            return LogisticRegression(random_state=42)
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Args:
            X: Input features
            
        Returns:
            array: Predicted labels
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
    
    def save_model(self, model_path):
        """
        Save the trained model
        
        Args:
            model_path (str): Path to save the model
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
    
    def load_model(self, model_path):
        """
        Load a trained model
        
        Args:
            model_path (str): Path to the saved model
        """
        self.model = joblib.load(model_path)
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance scores
        
        Args:
            feature_names (list): List of feature names
            
        Returns:
            dict: Dictionary mapping feature names to importance scores
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            return None
        
        return dict(zip(feature_names, importances)) 