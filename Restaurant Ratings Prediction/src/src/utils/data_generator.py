import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RestaurantDataGenerator:
    def __init__(self):
        self.cuisine_types = [
            'Italian', 'Japanese', 'Indian', 'American', 'Mexican', 
            'Chinese', 'French', 'Thai', 'Mediterranean', 'Korean'
        ]
        self.locations = [
            'Downtown', 'Suburb', 'City Center', 'Business District',
            'Shopping Mall', 'Tourist Area', 'Residential Area'
        ]
        
    def generate_restaurants(self, n_restaurants=100):
        """
        Generate mock restaurant data
        
        Args:
            n_restaurants (int): Number of restaurants to generate
            
        Returns:
            pd.DataFrame: Generated restaurant data
        """
        np.random.seed(42)
        
        data = {
            'restaurant_id': range(n_restaurants),
            'name': [f'Restaurant_{i}' for i in range(n_restaurants)],
            'cuisine_type': np.random.choice(self.cuisine_types, n_restaurants),
            'price_range': np.random.randint(1, 6, n_restaurants),
            'location': np.random.choice(self.locations, n_restaurants),
            'opening_year': np.random.randint(2000, 2024, n_restaurants)
        }
        
        return pd.DataFrame(data)
    
    def generate_users(self, n_users=1000):
        """
        Generate mock user data
        
        Args:
            n_users (int): Number of users to generate
            
        Returns:
            pd.DataFrame: Generated user data
        """
        np.random.seed(42)
        
        data = {
            'user_id': range(n_users),
            'age': np.random.randint(18, 71, n_users),
            'gender': np.random.choice(['M', 'F'], n_users),
            'location': np.random.choice(self.locations, n_users)
        }
        
        return pd.DataFrame(data)
    
    def generate_ratings(self, restaurants_df, users_df, n_ratings=10000):
        """
        Generate mock rating data
        
        Args:
            restaurants_df (pd.DataFrame): Restaurant data
            users_df (pd.DataFrame): User data
            n_ratings (int): Number of ratings to generate
            
        Returns:
            pd.DataFrame: Generated rating data
        """
        np.random.seed(42)
        
        # Generate random user-restaurant pairs
        user_ids = np.random.choice(users_df['user_id'], n_ratings)
        restaurant_ids = np.random.choice(restaurants_df['restaurant_id'], n_ratings)
        
        # Generate ratings with some bias based on price range and location
        ratings = []
        dates = []
        
        for i in range(n_ratings):
            restaurant = restaurants_df.loc[
                restaurants_df['restaurant_id'] == restaurant_ids[i]
            ].iloc[0]
            
            # Base rating with some randomness
            base_rating = np.random.normal(3.5, 0.5)
            
            # Adjust based on price range (higher price -> slightly higher rating)
            price_factor = (restaurant['price_range'] - 3) * 0.1
            
            # Adjust based on location match
            user_location = users_df.loc[
                users_df['user_id'] == user_ids[i], 'location'
            ].iloc[0]
            location_factor = 0.2 if restaurant['location'] == user_location else 0
            
            # Calculate final rating
            rating = base_rating + price_factor + location_factor
            rating = max(1, min(5, round(rating, 1)))  # Clamp between 1 and 5
            ratings.append(rating)
            
            # Generate random date within last 2 years
            days_ago = np.random.randint(0, 365*2)
            date = datetime.now() - timedelta(days=days_ago)
            dates.append(date)
        
        data = {
            'user_id': user_ids,
            'restaurant_id': restaurant_ids,
            'rating': ratings,
            'date': dates,
            'service_rating': np.random.randint(1, 6, n_ratings)
        }
        
        return pd.DataFrame(data)
    
    def generate_complete_dataset(self, n_restaurants=100, n_users=1000, 
                                n_ratings=10000, save_path=None):
        """
        Generate complete mock dataset
        
        Args:
            n_restaurants (int): Number of restaurants
            n_users (int): Number of users
            n_ratings (int): Number of ratings
            save_path (str, optional): Path to save the generated data
            
        Returns:
            tuple: (restaurants_df, users_df, ratings_df)
        """
        restaurants_df = self.generate_restaurants(n_restaurants)
        users_df = self.generate_users(n_users)
        ratings_df = self.generate_ratings(restaurants_df, users_df, n_ratings)
        
        if save_path:
            restaurants_df.to_csv(f"{save_path}/restaurants.csv", index=False)
            users_df.to_csv(f"{save_path}/users.csv", index=False)
            ratings_df.to_csv(f"{save_path}/ratings.csv", index=False)
        
        return restaurants_df, users_df, ratings_df

if __name__ == "__main__":
    # Example usage
    generator = RestaurantDataGenerator()
    restaurants, users, ratings = generator.generate_complete_dataset(
        save_path="data"
    ) 