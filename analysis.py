import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")


class Vizualiser:

    
    # cost distirboution
    def cost_dist(self,df):
        plt.figure(figsize=(10,6))
        sns.histplot(df['Cost'],kde=True)
        plt.title('Distribution of Hotel Costs')
        plt.xlabel("Cost")

    def top_10_cuisines(self,df):
        # spliting the cusines and storing in list 
        cuisine_value_list = df.Cuisines.str.split(', ')
        # storing all the cusines in a dict 
        cuisine_dict = {}
        for cuisine_names in cuisine_value_list:
            for cuisine in cuisine_names:
                if (cuisine in cuisine_dict):
                    cuisine_dict[cuisine]+=1
                else:  
                    cuisine_dict[cuisine]=1 
        # converting the dict to a data frame 
        cuisine_df=pd.DataFrame.from_dict([cuisine_dict]).transpose().reset_index().rename(
            columns={'index':'Cuisine',0:'Number of Restaurants'})
        top_10 = cuisine_df.sort_values(by='Number of Restaurants',ascending=False).head(10)
        plt.figure(figsize=(10,6))
        sns.barplot(x=top_10['Cuisine'],y=top_10['Number of Restaurants'])
        plt.title("Top 10 Cuisines")
        plt.xlabel("Cuisine")
        plt.ylabel("Count")

    def top_10_hotel(self,df):
        df['Rating'] = pd.to_numeric(df['Rating'],errors='coerce')
        # calculating the avg rating and the no of reviews they got
        avg_hotel_rating  = df.groupby(by='Restaurant').agg({"Rating":"mean","Reviewer":"count"}).reset_index().rename({"Reviewer":"Total_Review"})
        top_10_hotel = avg_hotel_rating.sort_values('Rating',ascending=False).head(10)
        plt.figure(figsize=(12,8))
        sns.barplot(y=top_10_hotel['Restaurant'],x=top_10_hotel['Rating'])
        plt.title("Top 10 Hotel Based on the rating")
        plt.xlabel("Rating")
        plt.ylabel("Hotel Name")

    def food_price_of_top_10_hotel(self,df):

        #Price point of restaurants
        price_point = df.groupby('Restaurant').agg({'Rating':'mean',
                'Cost': 'mean'}).reset_index().rename(columns = {'Cost': 'Price_Point'})
        # price point of top 10 resturent
        price_of_top_10_hotel = price_point.sort_values(by='Rating',ascending=False).head(10)
        plt.figure(figsize=(10,6))
        sns.barplot(y=price_of_top_10_hotel['Restaurant'],x=price_of_top_10_hotel['Price_Point'])
        plt.title("Top 10 Cuisines")
        plt.xlabel("Food price")
        plt.ylabel("Hotel Name")
        plt.show()

    def top_5_most_selling_food(self,df):
        #hotel
        # spliting the cusines and storing in list 
        Collections_value_list = df.Collections.dropna().str.split(', ')
        # storing all the cusines in a dict 
        Collections_dict = {}
        for collection in Collections_value_list:
            for col_name in collection:
                if (col_name in Collections_dict):
                    Collections_dict[col_name]+=1
                else:  
                    Collections_dict[col_name]=1 
                    # converting the dict to a data frame 
        Collections_df=pd.DataFrame.from_dict([Collections_dict]).transpose().reset_index().rename(
            columns={'index':'Tags',0:'Number of Restaurants'})
        collection_list = Collections_df.sort_values('Number of Restaurants', 
                                ascending = False)['Tags'].tolist()[:5]
        data = Collections_df.sort_values('Number of Restaurants', 
                                ascending = False)['Number of Restaurants'].tolist()[:5]

        plt.pie(data,labels=collection_list,autopct='%.0f%%')
        plt.title("Top 5 Most selling Food")
        plt.show()