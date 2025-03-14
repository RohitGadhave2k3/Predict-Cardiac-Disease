import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd

class Visualizer:
    @staticmethod
    def plot_rating_distribution(ratings, save_path=None):
        """
        Plot the distribution of ratings
        
        Args:
            ratings (array-like): Array of ratings
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(ratings, bins=20)
        plt.title('Distribution of Restaurant Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            labels (list, optional): Label names
            save_path (str, optional): Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if labels:
            plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45)
            plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba, save_path=None):
        """
        Plot ROC curve
        
        Args:
            y_true (array-like): True labels
            y_pred_proba (array-like): Predicted probabilities
            save_path (str, optional): Path to save the plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_feature_importance(feature_importance, feature_names, save_path=None):
        """
        Plot feature importance
        
        Args:
            feature_importance (array-like): Feature importance scores
            feature_names (list): Feature names
            save_path (str, optional): Path to save the plot
        """
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, y='feature', x='importance')
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_learning_curve(train_scores, test_scores, save_path=None):
        """
        Plot learning curve
        
        Args:
            train_scores (array-like): Training scores
            test_scores (array-like): Testing scores
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_scores) + 1), train_scores, 
                label='Training Score', marker='o')
        plt.plot(range(1, len(test_scores) + 1), test_scores,
                label='Testing Score', marker='o')
        plt.title('Learning Curve')
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show() 