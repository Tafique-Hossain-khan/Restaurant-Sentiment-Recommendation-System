import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis import Vizualiser
from sentiment_analysis import Sentiment

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity


from recommend import Recommend


# Load Dataset
hotel_df = pd.read_csv('Data/Zomato_Restaurant_names_and_Metadata.csv')
review_df = pd.read_csv('Data/Zomato_Restaurant_reviews.csv')
merge_df = pd.read_csv("Data/final_df.csv")
recommend_df = pd.read_csv('Data/recommend_df.csv')
with st.sidebar:
    selected = option_menu(
        'Restaurant App',
        ['Insights Dashboard', 'Sentiment Analysis', 'Personalized Restaurant Recommendations'],
        menu_icon="shop",  # Set a menu icon, such as a restaurant-related icon
        icons=['bar-chart', 'bezier2', 'star'],  # Use different icons from FontAwesome or Bootstrap
        default_index=0,
    )

# Depending on what is selected, display content
if selected == 'Insights Dashboard':
    st.title("Insights Dashboard")
    # Add your dashboard code here

    vis = Vizualiser()
    st.subheader("Cost Distribution")
    vis.cost_dist(merge_df)
    st.pyplot(plt)
    st.write("""This histogram shows the distribution of hotel costs, with most hotels priced between 500 and 1000. The line graph overlay highlights the same trend, indicating that the majority of hotels fall within the lower to mid-range cost category.""")

    #top10
    st.subheader("Top 10 Cuisines")
    vis.top_10_cuisines(hotel_df)
    st.pyplot(plt)

    #top_10_hotel base on rating
    st.subheader("Top 10 Hotel Based on the rating")
    vis.top_10_hotel(review_df)
    st.pyplot(plt)

    # price point of top 10 resturent
    st.subheader("Food Price Of Top 10 Hotel")
    vis.food_price_of_top_10_hotel(merge_df)
    st.pyplot(plt)


    #top 5 selling food
    st.subheader("Top 5 Selling Food")
    vis.top_5_most_selling_food(hotel_df)
    st.pyplot(plt)

    st.title("Hypothesis Testing Insights")

    # Hypothesis Testing
    #----------------------------

    # Introduction
    st.markdown("## Hypothesis Testing Results")
    st.write("""
    Below are the results of multiple hypothesis tests conducted to analyze various factors influencing restaurant ratings. 
    Each test presents the null and alternative hypotheses, p-values, and conclusions.
    """)

    # Function to display each test in a clean format
    def display_test(title, null_hypothesis, alt_hypothesis, p_value, result):
        col1, col2 = st.columns([3, 1])  # Create columns for better formatting
        with col1:
            st.markdown(f"### {title}")
            st.write(f"**Null Hypothesis (H₀)**: {null_hypothesis}")
            st.write(f"**Alternative Hypothesis (H₁)**: {alt_hypothesis}")
            st.write(f"**p-value**: {p_value}")
        with col2:
            # Use badges to highlight result
            if "Reject" in result:
                st.markdown(f"<span style='color:green'>✔ {result}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:red'>✘ {result}</span>", unsafe_allow_html=True)

    # Display each hypothesis test result
    display_test(
        "Q1: Cost vs Rating",
        "No relationship between 'Cost' and 'Rating'.",
        "Significant relationship between 'Cost' and 'Rating'.",
        4.595644071498114e-47,
        "Reject Null Hypothesis"
    )

    display_test(
        "Q2: Number of Followers vs Rating ",
        "The number of followers has no effect on rating.",
        "More followers lead to higher ratings.",
        4.595644071499624e-47,
        "Reject Null Hypothesis"
    )

    display_test(
        "Q3: Number of Followers vs Rating ",
        "The number of followers has no effect on rating.",
        "More followers lead to higher ratings.",
        0.0004760318830883612,
        "Reject Null Hypothesis"
    )

    display_test(
        "Q4: Variety of Cuisines vs Rating",
        "Cuisine variety has no effect on restaurant rating.",
        "A wider variety of cuisines positively affects rating.",
        0.0,
        "Reject Null Hypothesis"
    )


    # Conclusion
    st.write("""
    ### Conclusion:
    Based on the tests, we reject the null hypotheses in all cases, suggesting significant relationships between the analyzed factors and restaurant ratings. These insights could help shape strategies to improve restaurant performance.
    """)




elif selected == 'Personalized Restaurant Recommendations':


    # Title and subtitle
    st.title("Restaurant Recommendation System")
    st.markdown("### Get personalized recommendations based on your preferences!")

    # Step 1: Choose recommendation type
    st.markdown("<h2 style='color: #1f77b4;'>Choose the Recommendation Method:</h2>", unsafe_allow_html=True)

    # Styling for the radio buttons using a custom layout
    recommendation_type = st.radio(
        "",
        ('Based on Review Similarity', 'Based on Cuisine and Rating'),
        index=0,
        horizontal=True
    )

    # Style adjustments for radio buttons
    st.markdown("""
        <style>
            .stRadio {
                display: flex;
                justify-content: space-around;
                padding: 20px 0;
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                background-color: #f9f9f9;
            }
            .stRadio label {
                font-size: 18px;
                font-weight: bold;
                color: #333;
                cursor: pointer;
                transition: transform 0.2s;
                padding: 10px 20px;
                border-radius: 5px;
            }
            .stRadio input[type="radio"] {
                display: none;  /* Hide default radio button */
            }
            .stRadio input[type="radio"]:checked + label {
                background-color: #1f77b4;
                color: white;
                transform: scale(1.05);
            }
            .stRadio label:hover {
                background-color: #e0e0e0;
            }
        </style>
    """, unsafe_allow_html=True)


    # Step 2: Handle selection based on the user's choice
    rec = Recommend()
    if recommendation_type == 'Based on Review Similarity':
        # Review similarity option
        hotel_list = recommend_df['Restaurant'].unique()
        selected_restaurant = st.selectbox("Select a restaurant for review-based recommendations:", 
                                        hotel_list)
        
        # Button to trigger recommendations
        if st.button("Get Recommendations"):
            recommendations = rec.get_review_similarity_recommendations(recommend_df,selected_restaurant)
            
            hotel_name = []
            
            for hotel in recommendations:
                hotel_name.append(hotel[0])

            if not recommendations:
                st.error("No similar restaurants found based on reviews.")
            else:
                st.success(f"✨ Restaurants similar to {selected_restaurant} based on customer reviews:")

                # Display the recommendations in a visually appealing box format
                for hotel in hotel_name:
                    st.markdown(f"""
                        <div style="background-color:#f9f9f9;padding:10px;margin-bottom:10px;border-radius:8px;border: 1px solid #ddd;">
                            <h3 style="color:#007acc;">{hotel}</h3>
                            
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("### Enjoy your meal at these similar places! 🍽️")

    elif recommendation_type == 'Based on Cuisine and Rating':
        # Cuisine and rating option
        cuisine_list = ['American', 'Andhra', 'Arabian',
       'Asian', 'BBQ', 'Bakery', 'Beverages', 'Biryani', 'Burger', 'Cafe',
       'Chinese', 'Continental', 'Desserts', 'European', 'Fast Food',
       'Finger Food', 'Goan', 'Healthy Food', 'Hyderabadi', 'Ice Cream',
       'Indonesian', 'Italian', 'Japanese', 'Juices', 'Kebab', 'Lebanese',
       'Malaysian', 'Mediterranean', 'Mexican', 'Mithai', 'Modern Indian',
       'Momos', 'Mughlai', 'North Eastern', 'North Indian', 'Pizza', 'Salad',
       'Seafood', 'South Indian', 'Spanish', 'Street Food', 'Sushi', 'Thai',
       'Wraps']
        user_cuisines = st.multiselect("Select cuisines:", 
                                    cuisine_list)   
        rating_threshold = st.slider("Select minimum rating threshold:", 1.0, 5.0, 4.0)

        # Button to trigger recommendations
        if st.button("Get Recommendations"):
            if not user_cuisines:
                st.warning("Please select at least one cuisine.")
            else:
                recommendations = rec.cuisine_based_recommendation(recommend_df,hotel_df, rating_threshold, user_cuisines)
                
                if not recommendations:
                    st.error("No restaurants match your criteria.")
                else:
                    st.success(f"✨ Restaurants offering {', '.join(user_cuisines)} with a rating above {rating_threshold}:")

                    # Display the recommendations in a visually appealing box format
                    for hotel in recommendations:
                        st.markdown(f"""
                            <div style="background-color:#f9f9f9;padding:10px;margin-bottom:10px;border-radius:8px;border: 1px solid #ddd;">
                                <h3 style="color:#ff6347;">{hotel}</h3>
                                
                            </div>
                        """, unsafe_allow_html=True)

                    st.markdown("### Enjoy your meal at these similar places! 🍽️")


elif selected == 'Sentiment Analysis':
        df = pd.read_csv("Data\\sentiment_tag.csv")

        # Streamlit app
        st.title('Restaurant Review Sentiment Analysis with Preprocessing and Sentiment')

        # Dropdown to select restaurant
        restaurant = st.selectbox('Select Restaurant', df['Restaurant'].unique())

        # Filter data based on selected restaurant
        filtered_df = df[df['Restaurant'] == restaurant]
        sentiment_counts = filtered_df['Sentiment'].value_counts()

        st.subheader('Sentiment Breakdown')

        # Get the actual sentiments present in the filtered data
        actual_labels = sentiment_counts.index.tolist()

        # Map of all possible labels and their corresponding colors
        label_color_map = {
            'Positive': '#66b3ff',  # Blue
            'Neutral': '#99ff99',   # Green
            'Negative': '#ff6666'   # Red
        }

        # Filter the label_color_map to only include the actual sentiments
        labels = [label for label in ['Positive', 'Neutral', 'Negative'] if label in actual_labels]
        colors = [label_color_map[label] for label in labels]

        # Adjust explode based on the number of sentiments (only explode Positive if present)
        explode = [0.1 if label == 'Positive' else 0 for label in labels]

        # Create the pie chart
        fig, ax = plt.subplots(figsize=(8, 6))  # Increase figure size
        ax.pie(sentiment_counts, 
            labels=labels, 
            autopct='%0.1f%%', 
            explode=explode, 
            shadow=True, 
            colors=colors, 
            startangle=90,  # Adjusts the start angle of the chart
            wedgeprops={'edgecolor': 'black'},  # Adds edge color for slices
            textprops={'fontsize': 12},  # Adjusts text style
            pctdistance=0.85,  # Adjust percentage distance from center
            labeldistance=1.1)  # Adjust label distance from center

        # Add a title
        ax.set_title('Sentiment Analysis', fontsize=20, color='darkblue')

        # Add a legend outside the pie chart
        ax.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)

        # Ensure equal aspect ratio for a perfect circle
        ax.axis('equal')

        # Display the pie chart in Streamlit
        st.pyplot(fig)
