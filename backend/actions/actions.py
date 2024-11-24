# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions
# pip install sketch
import datetime as dt
import string
import random
# import openai
# pip install openai
from typing import Any, Text, Dict, List
import pandas as pd
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
# openai.api_key = 'your_openai_api_key'
import requests
# import docx2txt
import platform
import sqlalchemy
from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import KNNBasic
import random
import csv
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

class Astros(Action):
    def name(self) -> Text:
        return "action_astros"

    def run(self, dispatcher, tracker, domain):
        #       api_key = ''
        astr_name_ = tracker.get_slot('astr_name')
        current = requests.get("http://api.open-notify.org/astros.json")
        x = current.json()
        dataframe = pd.DataFrame(x)
        new_df = dataframe["people"].apply(pd.Series)
        craft = new_df[new_df['name'] == astr_name_]['craft'].values[0]
        response = """{} is in {} at the moment.""".format(astr_name_, craft)
        dispatcher.utter_message(response)
        return [SlotSet('astr_name', astr_name_)]


class ActionEnquireAirportData(Action):

    def name(self) -> Text:
        return "action_ask_about_airport_data"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data_ = pd.read_excel(
            r"C:\Users\Shivam Dutt Sharma\Desktop\NZ_Conversational_AI\Auckland_Monthly_Traffic_Statistics_Jan2013_to_July2023.xlsx")
        data_df = pd.DataFrame(data_)
        find_ = tracker.get_slot('find')
        attribute_ = tracker.get_slot('attribute')
        feature_ = tracker.get_slot('feature')
        column_names = list(data_df.columns.values)
        for _ in column_names:
            if _ in feature_:
                feature_ = _
                break
        if find_ in ('maximum', 'max', 'highest', 'largest') and attribute_ in ('month', 'Month', 'MONTH'):
            val_attr = data_df[data_df[feature_] == data_df[feature_].max()]['Month'].to_string(index=False)
            val_find_feature = data_df[data_df[feature_] == data_df[feature_].max()][feature_].to_string(index=False)
            # also try just data_df[feature_].max().to_string(index=False)
            dispatcher.utter_message(
                text=f"Over the last 10 years in Auckland, {val_find_feature} was the {find_} {feature_} that "
                     f"happened in {val_attr}.")
        elif find_ in ('minimum', 'min', 'lowest', 'smallest') and attribute_ in ('month', 'Month', 'MONTH'):
            val_attr = data_df[data_df[feature_] == data_df[feature_].min()]['Month'].to_string(index=False)
            val_find_feature = data_df[data_df[feature_] == data_df[feature_].min()][feature_].to_string(index=False)
            # also try just data_df[feature_].min().to_string(index=False)
            dispatcher.utter_message(
                text=f"Over the last 10 years in Auckland, {val_find_feature} was the {find_} {feature_} that "
                     f"happened in {val_attr}.")
        else:
            val_find_feature = data_df[feature_].mean().to_string(index=False)
            dispatcher.utter_message(
                text=f"Over the last ten years, on an {find_} Auckland experienced {val_find_feature} {feature_}")

        return []

class Correlations(Action):
    def name(self) -> Text:
        return "action_correlations"

    def run(self, dispatcher, tracker, domain):
        #       api_key = ''
        dataset_ = tracker.get_slot('dataset')
        # Create a sample DataFrame
        data = pd.read_csv(r"C:\Users\Shivam Dutt Sharma\Desktop\Prajna\ConversationalAIPrajna\Datasets\ICC Cricket World Cup 2023.csv")
        df = pd.DataFrame(data)
        # Calculate the correlation matrix
        correlation_matrix = df.corr()
        cor_mat_df = pd.DataFrame(correlation_matrix)
        cor_mat_df.drop(list(cor_mat_df.filter(regex='Unnamed')), axis=1, inplace=True)
        cor_mat_df = cor_mat_df.dropna()
        mask = 1 - pd.DataFrame(numpy.eye(len(cor_mat_df)), index=cor_mat_df.index, columns=cor_mat_df.columns)
        cor_mat_df = cor_mat_df * mask
        max_column = cor_mat_df.idxmax(axis=1)
        max_value = cor_mat_df.max(axis=1)
        result_df = pd.DataFrame({'Max_Column': max_column, 'Max_Value': max_value})
        return_str = ""
        for _, _maxcol, _maxval in result_df.itertuples():
            # print(_,_maxcol, _maxval, "\n")
            # print(type(_maxcol))
            return_str += "The {} feature has the highest correlation of {} with {}.  ".format(_, _maxval, _maxcol)
        # Print the correlation matrix
        # print(correlation_matrix)
        # cor_mat_df = pd.DataFrame(correlation_matrix)
        # Create a heatmap of the correlation matrix
        # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        # image_path = r'C:\Users\Shivam Dutt Sharma\Desktop\Prajna\ConversationalAIPrajna\plot.png'
        # plt.savefig(image_path)
        # plt.close()
        # image_url = f'http://192.168.182.106:5002/plot_image'
        # response_ = f"Here's the correlations plot"
        # dispatcher.utter_message(text=f"{response_} {image_url}")
        dispatcher.utter_message(text=f"{return_str}")
        return []
class ICCCricketWorldCup2023(Action):
    def name(self) -> Text:
        return "action_general_insights_cricket_worldcup_2023"

    def run(self, dispatcher, tracker, domain):
        #       api_key = ''

        find_slot = tracker.get_slot('find_')
        country_slot = tracker.get_slot('country_')
        data_ = pd.read_csv(r"C:\Users\Shivam Dutt Sharma\Desktop\Prajna\NZ_Conversational_AI\WT\datasets\data_icc_cricket_world_cup.csv")
        data_df = pd.DataFrame(data_)
        # data_df
        data_df['match_date'] = pd.to_datetime(data_df['match_date'])
        icc_cricket_world_cup_start_date = pd.to_datetime('10/8/2023')
        data_df = data_df[data_df['match_date'] > icc_cricket_world_cup_start_date]
        if find_slot in ('max', 'best', 'maximum'):
            max_runs_df = data_df.query("india_wins==1 & margin_label == 'runs' & india_runs==india_runs.max()")
            max_runs_opponent = max_runs_df.opponent.to_string(index=False)
            max_runs_margin = max_runs_df.margin.to_string(index=False)
            max_runs_date = max_runs_df.match_date.to_string(index=False)
            max_runs_runs_scored = max_runs_df.india_runs.to_string(index=False)
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------
            max_wickets_df = data_df.query("india_wins==1 &  margin_label == 'wickets'")
            max_wickets_df = max_wickets_df.query("margin==margin.max()")
            max_wickets_opponent = max_wickets_df.opponent.to_string(index=False)
            max_wickets_margin = max_wickets_df.margin.to_string(index=False)
            # max_runs_margin
            max_wickets_date = max_wickets_df.match_date.to_string(index=False)

        response_ = f"In this ICC Cricket World Cup 2023, India played against {max_runs_opponent} on {max_runs_date} and scored a whooping {max_runs_runs_scored}, winning by {max_runs_margin} runs. This was the best batting performance from India. When it comes to India's bowling side, it showed an awe-inspiring performance on {max_wickets_date}, when it defeated {max_wickets_opponent} by a stunning {max_wickets_margin} wickets."
        dispatcher.utter_message(response_)
        return []


class ActionEnquireNews(Action):
    def name(self) -> Text:
        return "action_enquire_news"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response_ = requests.get(
            "https://newsapi.org/v2/everything?q=tesla&from=2023-02-02&sortBy=publishedAt&apiKey"
            "=1814daf764864140bc41fc7b04119539")
        df_ = pd.DataFrame(response_.json()['articles'])
        desc_ = df_['description'][0]

        dispatcher.utter_message(text=f"Here is one of the articles related to Tesla : \n {desc_}")
        return []

# class ActionPlayAGame(Action):
#     def name(self) -> Text:
#         return "action_play_a_game"
#     def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

class ActionEnquireAboutAnything(Action):
    import wikipediaapi
    import re
    def name(self) -> Text:
        return "action_enquire_about_anything"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        import wikipediaapi
        import re
        wiki_wiki = wikipediaapi.Wikipedia('en')
        x_ = tracker.get_slot('entity_name')
        page_py = wiki_wiki.page(x_)
        page_missing = wiki_wiki.page('NonExistingPageWithStrangeName')
        result_string = page_py.summary[0:600].partition(' is')[2]
        return_text = x_ + " is" + ' '.join(re.split(r"(?<=[.:;])\s", result_string)[:3])
        dispatcher.utter_message(text=return_text)
        return []
class EDA(Action):
    def name(self) -> Text:
        return "action_do_eda_of_this_dataset"

    def run(selfself, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dataset_ = tracker.get_slot('dataset')
        if dataset_ == '1':
            dataset_ = ''

class ActionRecommendations(Action):
    def name(self) -> Text:
        return "action_give_me_movie_recommendations"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_ = tracker.get_slot('user')
        if user_:
            has_have = 'has'
        else:
            user_ = 'you'
            has_have = 'have'
        reader = Reader(line_format='user item rating timestamp', sep='\t', rating_scale=(1, 5))
        data = Dataset.load_builtin('ml-100k')

        # Build the full training set
        # trainset = data.build_full_trainset()
        trainset, testset = train_test_split(data, test_size=0.2)

        # Use user-based collaborative filtering
        sim_options = {
            'name': 'cosine',
            'user_based': True
        }

        model = KNNBasic(sim_options=sim_options, min_k=1)
        model.fit(trainset)

        # Specify the target user for recommendation
        #n = random.randint(1, 1)
        # print(n)
        #target_user_id = str(n)
        target_user_id = str(random.randint(1, trainset.n_users))

        # Get a list of movie IDs that the target user hasn't rated
        movies_not_rated = [movie_id for movie_id in trainset.all_items() if
                            trainset.ur[trainset.to_inner_uid(target_user_id)]]

        # Predict ratings for movies not rated by the target user
        predictions = [model.predict(target_user_id, movie_id) for movie_id in movies_not_rated]

        # Create a mapping between inner and raw item IDs
        inner_to_raw_items = {inner_id: raw_id for raw_id, inner_id in trainset._raw2inner_id_items.items()}

        # Recommend the top N movies with the highest predicted ratings
        top_n = 10
        recommended_movies = sorted(predictions, key=lambda x: x.est, reverse=True)[:top_n]

        # Display the recommended movie names
        # print(
        #     "Looking at the types of movies you have liked in the past and the preferences of other users who are similar to you, here are a few recommendations from me:")
        recommended_movies_list = []
        for movie in recommended_movies:
            raw_movie_id = inner_to_raw_items.get(movie.iid, "Unknown Movie")

            with open(r"C:\Users\Shivam Dutt Sharma\Desktop\Prajna_Github\prajna.github.io\backend\datasets\movies.csv", encoding="utf8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if (row['movieId'] == raw_movie_id):
                        recommended_movies_list.append(row['title'])
        response_ = f"Looking at the types of movies {user_} {has_have} liked in the past and the preferences of other users who are similar to {user_}, here are a few movie recommendations from me for {user_}: {recommended_movies_list}."
        dispatcher.utter_message(response_)
        return []
# def generate_chatgpt_response(prompt):
#     response = openai.Completion.create(
#         engine="text-davinci-002",  # Specify the GPT-3 engine
#         prompt=prompt,
#         max_tokens = 150  # Adjust based on desired response length
#     )
#     return response['choices'][0]['text'].strip()

class ActionFallback(Action):
    def name(self) -> Text:
        return "action_fallback"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_input = tracker.latest_message.get("text")
        # Call ChatGPT to generate a response
        chatgpt_response = generate_chatgpt_response(user_input)
        dispatcher.utter_message(chatgpt_response)
        return []

class ActionPerformCorrelations(Action):   
    def name(self) -> Text:
        return "action_perform_correlations"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        import pandas as pd
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        dataset_ = tracker.get_slot('dataset_name')
        if dataset_ == 'john_wick_daisy_investigation':
           dataset_ = pd.read_csv('C:\\Users\\Shivam Dutt Sharma\\Desktop\\Desktop _ Mar 04 2024\\Prajna\\ConversationalAIPrajna\\WT\\datasets\\keanu_reaves_data.csv')
        else:
           dataset_ = 'C:\\Users\\Shivam Dutt Sharma\\Desktop\\Desktop _ Mar 04 2024\\Prajna\\ConversationalAIPrajna\\WT\\datasets\\data_icc_cricket_world_cup.csv'

        dataset_df = pd.DataFrame(dataset_)
        num_cols = dataset_df.select_dtypes(include = 'number').columns.tolist()
        correlation_matrix = dataset_df[num_cols].corr()

        correlation_threshold = 0

        # Find highly correlated columns
        highly_correlated_pairs = []
        highly_correlated_pairs_txt = ''
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                    highly_correlated_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

        # Print highly correlated pairs

        for pair in highly_correlated_pairs:
            highly_correlated_pairs_txt += str(pair) + " "
        corr_text = f"Highly correlated pairs identified from correlation matrix are : {highly_correlated_pairs_txt}"

        # Now, let's perform VIF analysis
        # Assuming 'data' is your original DataFrame containing the columns
        # Replace 'data' with your DataFrame name

        # Create a DataFrame for VIF analysis excluding any non-numeric columns
        numeric_data = dataset_df.select_dtypes(include=['float64', 'int64'])

        # Calculate VIF for each column
        vif_data = pd.DataFrame()
        vif_data["Feature"] = numeric_data.columns
        vif_data["VIF"] = [variance_inflation_factor(numeric_data.values, i) for i in range(len(numeric_data.columns))]

        # Set threshold for high VIF
        vif_threshold = 5

        # Find columns with high VIF
        high_vif_columns = vif_data[vif_data["VIF"] > vif_threshold]["Feature"].tolist()

        # Print columns with high VIF
        vif_text = f"\nColumns with high VIF identified from VIF analysis are : {high_vif_columns}"
        # print(high_vif_columns)
        # corr_return_text = f"These variables have a very high correlation. I kept the threshold as 0.5. So you can take these as variables which have really high correlations - {variables_with_high_correlations}."
        return_text = corr_text + vif_text
        dispatcher.utter_message(return_text)
        return []
