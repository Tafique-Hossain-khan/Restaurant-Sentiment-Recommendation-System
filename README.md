
# Restaurant Sentiment Recommendation System

**[Live Web App](https://restaurant-sentiment-recommendation-system-4auqz5fk8g5lcbgyafp.streamlit.app/)**

## Overview
The **Restaurant Sentiment Recommendation System** is a web application that analyzes restaurant reviews and recommends restaurants based on sentiment analysis, cuisine preferences, and ratings. This project applies **Natural Language Processing (NLP)** techniques to understand customer feedback and uses a **content-based recommendation system** to provide personalized suggestions.

The system utilizes **cosine similarity** to compare reviews, allowing users to discover restaurants with similar reviews and sentiments. Additionally, it offers filtering based on cuisine and ratings, enhancing the recommendation process.

## Features
- **Sentiment Analysis on Reviews:**  
  Analyzes restaurant reviews using the `SentimentIntensityAnalyzer` from **NLTK** to determine sentiment polarity (positive, negative, neutral).
  
- **Content-Based Recommendations:**  
  Recommends restaurants by calculating **cosine similarity** on the text of reviews, helping users find restaurants with similar feedback. 
  
- **Cuisine and Rating Filters:**  
  Users can filter recommendations based on specific cuisine types and restaurant ratings, enabling more personalized and relevant results.

- **Clustering for Enhanced Recommendations:**  
  Groups restaurants based on their features (e.g., reviews, ratings, cuisine) to improve the recommendation process.

## How It Works
1. **Data Preprocessing**:  
   Reviews are cleaned and tokenized using **NLTK** to prepare for sentiment analysis and similarity computation.
   
2. **Sentiment Analysis**:  
   Sentiment scores are computed for each review using the **SentimentIntensityAnalyzer**, which classifies them into positive, negative, or neutral sentiments.
   
3. **Cosine Similarity for Recommendations**:  
   The system computes the cosine similarity between the reviews of different restaurants to identify those with similar review sentiments and patterns.
   
4. **Filtering by Cuisine and Ratings**:  
   Users can input their preferred cuisine and rating range, and the system filters recommendations accordingly.

## Tech Stack
- **Programming Language**: Python
- **Libraries**:
  - **NLTK**: For NLP tasks such as text preprocessing and sentiment analysis.
  - **Pandas**: For data manipulation and analysis.
  - **Scikit-learn**: For cosine similarity calculations.
  - **Streamlit**: To build and deploy the web interface.
  
## Installation & Setup
To run this project locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tafique-Hossain-khan/Restaurant-Sentiment-Recommendation-System.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd restaurant-sentiment-recommendation
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv env
   ```

4. **Activate the virtual environment**:
   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```
   - On MacOS/Linux:
     ```bash
     source env/bin/activate
     ```

5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## Usage
- Launch the web application.
- Select your preferred cuisine and rating filter.
- The app will recommend restaurants based on the sentiment of their reviews and the chosen filters.

## Future Enhancements
- Add **collaborative filtering** to improve recommendations.
- Implement **real-time data streaming** for continuously updated reviews.
- Enhance the UI/UX for a more interactive experience.
  
