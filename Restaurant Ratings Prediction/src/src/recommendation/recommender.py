import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

class RestaurantRecommender:
    def __init__(self):
        self.user_restaurant_matrix = None
        self.restaurant_features = None
        self.user_features = None
        
    def create_user_restaurant_matrix(self, ratings_df):
        """
        Create user-restaurant rating matrix
        
        Args:
            ratings_df (pd.DataFrame): DataFrame containing user ratings
            
        Returns:
            pd.DataFrame: User-restaurant matrix
        """
        self.user_restaurant_matrix = ratings_df.pivot(
            index='user_id',
            columns='restaurant_id',
            values='rating'
        ).fillna(0)
        return self.user_restaurant_matrix
    
    def train_collaborative_filtering(self, n_factors=50):
        """
        Train collaborative filtering model using SVD
        
        Args:
            n_factors (int): Number of latent factors
        """
        # Normalize the ratings
        ratings_mean = np.mean(self.user_restaurant_matrix, axis=1)
        ratings_demeaned = self.user_restaurant_matrix - ratings_mean.reshape(-1, 1)
        
        # Perform SVD
        U, sigma, Vt = svds(ratings_demeaned, k=n_factors)
        
        # Convert to diagonal matrix
        sigma = np.diag(sigma)
        
        # Store the features
        self.user_features = np.dot(U, np.sqrt(sigma))
        self.restaurant_features = np.dot(np.sqrt(sigma), Vt)
        
    def get_collaborative_recommendations(self, user_id, n_recommendations=5):
        """
        Get collaborative filtering based recommendations
        
        Args:
            user_id: ID of the user
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of recommended restaurant IDs
        """
        if user_id not in self.user_restaurant_matrix.index:
            return []
            
        user_idx = self.user_restaurant_matrix.index.get_loc(user_id)
        user_pred = np.dot(self.user_features[user_idx, :], self.restaurant_features)
        
        # Get restaurants that user hasn't rated
        user_ratings = self.user_restaurant_matrix.loc[user_id]
        unrated_restaurants = user_ratings[user_ratings == 0].index
        
        # Sort predictions and get top N
        pred_ratings = pd.Series(user_pred, index=self.user_restaurant_matrix.columns)
        pred_ratings = pred_ratings[unrated_restaurants]
        return pred_ratings.nlargest(n_recommendations).index.tolist()
    
    def train_content_based(self, restaurant_features_df):
        """
        Train content-based filtering model
        
        Args:
            restaurant_features_df (pd.DataFrame): DataFrame containing restaurant features
        """
        self.restaurant_features = restaurant_features_df
        self.restaurant_similarity = cosine_similarity(restaurant_features_df)
        
    def get_content_based_recommendations(self, restaurant_id, n_recommendations=5):
        """
        Get content-based filtering recommendations
        
        Args:
            restaurant_id: ID of the restaurant
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of recommended restaurant IDs
        """
        if restaurant_id not in range(len(self.restaurant_similarity)):
            return []
            
        # Get similarity scores
        sim_scores = self.restaurant_similarity[restaurant_id]
        
        # Sort restaurants by similarity
        sim_restaurants = pd.Series(sim_scores, 
                                  index=self.restaurant_features.index)
        
        # Exclude the input restaurant
        sim_restaurants = sim_restaurants.drop(restaurant_id)
        
        return sim_restaurants.nlargest(n_recommendations).index.tolist()
    
    def hybrid_recommendations(self, user_id, n_recommendations=5, 
                             collaborative_weight=0.7):
        """
        Get hybrid recommendations combining collaborative and content-based filtering
        
        Args:
            user_id: ID of the user
            n_recommendations (int): Number of recommendations to return
            collaborative_weight (float): Weight for collaborative filtering (0-1)
            
        Returns:
            list: List of recommended restaurant IDs
        """
        # Get collaborative filtering recommendations
        cf_recs = self.get_collaborative_recommendations(
            user_id, n_recommendations=n_recommendations*2)
        
        # Get content-based recommendations based on user's highly rated restaurants
        user_ratings = self.user_restaurant_matrix.loc[user_id]
        top_rated = user_ratings.nlargest(3).index.tolist()
        
        cb_recs = []
        for rest_id in top_rated:
            cb_recs.extend(self.get_content_based_recommendations(
                rest_id, n_recommendations=n_recommendations))
        
        # Combine recommendations
        cf_score = {rest: collaborative_weight * (n_recommendations*2 - i)
                   for i, rest in enumerate(cf_recs)}
        cb_score = {rest: (1-collaborative_weight) * (len(cb_recs) - i)
                   for i, rest in enumerate(cb_recs)}
        
        # Merge scores
        final_scores = {}
        for rest in set(cf_recs + cb_recs):
            final_scores[rest] = cf_score.get(rest, 0) + cb_score.get(rest, 0)
            
        # Sort and return top recommendations
        sorted_recs = sorted(final_scores.items(), 
                           key=lambda x: x[1], reverse=True)
        return [rest for rest, _ in sorted_recs[:n_recommendations]] 