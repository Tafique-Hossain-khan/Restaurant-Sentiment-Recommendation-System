from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class Sentiment:

    def __init__(self,df) -> None:
        self.df = df

    def analyze_sentiment(self,review):
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(review)
        if score['compound'] >= 0.05:
            return 'Positive'
        elif score['compound'] <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    def analyse(self,sentiment_df):
        sentiment_df['Review'] = sentiment_df['Review'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentiment_df['Review'])
        sentiment_df['Sentiment'] = sentiment_df['Review'].apply(self.analyze_sentiment)
        

        return sentiment_df
        
            