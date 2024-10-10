
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class Recommend:


    def cuisine_based_recommendation(self,recommend_df,hotel_df, rating_threshold, user_cuisines):
        # Filter DataFrame based on user cuisines and rating
        filtered_df = recommend_df[
            (recommend_df['Average_Rating'] >= rating_threshold) &
            (hotel_df['Cuisines'].apply(lambda x: any(cuisine in x for cuisine in user_cuisines)))
        ]

        # Sort by rating or other criteria if desired (optional)
        filtered_df = filtered_df.sort_values(by='Average_Rating', ascending=False)

        # Extract top 5 recommended restaurant names
        recommendations = filtered_df['Restaurant'].head(5).tolist()

        return recommendations
    

    def get_review_similarity_recommendations(self,recommend_df,selected_restaurant):
        # Generate TF-IDF vectors for the reviews
        tfidf = TfidfVectorizer(max_features=500)
        review_vectors = tfidf.fit_transform(recommend_df['Review']).toarray()

        # Compute cosine similarity between all reviews
        cosine_sim = cosine_similarity(review_vectors)

        # Get the index of the selected restaurant
        selected_index = recommend_df[recommend_df['Restaurant'] == selected_restaurant].index[0]

        # Get similarity scores for the selected restaurant, excluding the selected restaurant itself
        sim_scores = list(enumerate(cosine_sim[selected_index]))
        sim_scores = [(i, score) for i, score in sim_scores if i != selected_index]  # Exclude the selected restaurant

        # Sort the restaurants based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[:5]  # Get top 5 similar restaurants

        # Get restaurant names and similarity scores
        similar_restaurants = [(recommend_df.iloc[i]['Restaurant'], score) for i, score in sim_scores]
        
        return similar_restaurants
    

