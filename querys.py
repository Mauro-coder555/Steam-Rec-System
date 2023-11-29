
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel




# Load datasets


play_time_genre_df = pd.read_csv('simplified-data/play_time_genre.csv')
user_for_genre_df = pd.read_csv('simplified-data/user_for_genre.csv')
users_recommend_df = pd.read_csv('simplified-data/users_recommend.csv')
users_worst_developer_df = pd.read_csv('simplified-data/users_worst_developer.csv')
sentiment_analysis_df = pd.read_csv('simplified-data/sentiment_analysis.csv')
game_recomendation_df = pd.read_csv('simplified-data/game_recomendation.csv')



def PlayTimeGenre(genre: str) -> str:
    """
    Filter the DataFrame for the provided genre and find the year with the most playtime.

    Input:
    - genre (str): The genre for filtering the DataFrame.

    Output:
    - str: A string representation of a dictionary containing the year of release with the most playtime for the specified genre.
    """
    genre_df = play_time_genre_df[play_time_genre_df['genre'] == genre]
    max_playtime_index = genre_df['max_playtime_hours'].idxmax()
    year_with_max_playtime = genre_df.loc[max_playtime_index, 'year']
    return str({"Year of release with the most playtime for {} : {}".format(genre, year_with_max_playtime)})


def UserForGenre(genre: str) -> str:
    """
    Filter the DataFrame for the specified genre, find the user with the most played hours, and create a list of accumulated hours played per year.

    Input:
    - genre (str): The genre for filtering the DataFrame.

    Output:
    - str: A string representation of a dictionary containing the user with the most played hours for the specified genre and a list of accumulated hours per year.
    """
    df_genre = user_for_genre_df[user_for_genre_df['genres'] == genre]
    user_most_hours = df_genre.loc[df_genre['playtime_forever'].idxmax(), 'user']
    accumulation_hours_per_year = df_genre.groupby('year')['playtime_forever'].sum().reset_index()
    accumulation_hours_per_year = accumulation_hours_per_year.rename(columns={'year': 'Year', 'playtime_forever': 'Hours'})
    list_accumulation_hours = accumulation_hours_per_year.to_dict(orient='records')
    to_return = {
        "User with the most played hours for Genre {}".format(genre): user_most_hours,
        "Hours played": list_accumulation_hours
    }
    return str(to_return)


def UsersRecommend(year: int) -> str:
    """
    Filter the DataFrame for the provided year, sort it by the number of recommendations in descending order, and create a list of dictionaries with the game rankings.

    Input:
    - year (int): The year for filtering the DataFrame.

    Output:
    - str: A string representation of a list of dictionaries containing the game rankings for the specified year.
    """
    año_int = int(year)
    df_filtered = users_recommend_df[users_recommend_df['year'].astype(int) == año_int]
    df_sorted = df_filtered.sort_values(by='recommendations_count', ascending=False)
    resultado = [{"Puesto {}".format(i + 1): row['item_id']} for i, (_, row) in enumerate(df_sorted.iterrows())]
    return str(resultado)


def UsersWorstDeveloper(año: int) -> str:
    """
    Filter the DataFrame for the provided year, sort it by the least recommended count in ascending order, and create a list of dictionaries with the game rankings.

    Input:
    - año (int): The year for filtering the DataFrame.

    Output:
    - str: A string representation of a list of dictionaries containing the game rankings for the specified year.
    """
    año_int = int(año)
    df_filtered = users_worst_developer_df[users_worst_developer_df['year'].astype(int) == año_int]
    df_sorted = df_filtered.sort_values(by='least_recommended_count')
    resultado = [{"Puesto {}".format(i + 1): row['item_id']} for i, (_, row) in enumerate(df_sorted.iterrows())]
    return str(resultado)


def sentiment_analysis(empresa_desarrolladora: str) -> dict:
    """
    Filter the DataFrame for the specified developer, perform sentiment analysis, and return a dictionary of sentiment counts.

    Input:
    - empresa_desarrolladora (str): The developer's name for filtering the DataFrame.

    Output:
    - dict: A dictionary containing sentiment counts for the specified developer.
    """
    empresa_df = sentiment_analysis_df[sentiment_analysis_df['developer'] == empresa_desarrolladora]
    result_dict = {empresa_desarrolladora: []}
    sentiment_columns = ['Negative', 'Neutral', 'Positive']
    for sentiment in sentiment_columns:
        count = empresa_df[sentiment].values[0]
        result_dict[empresa_desarrolladora].append({sentiment: count})
    return result_dict


def recomendacion_juego(product_id: int) -> str:
    """
    Perform game recommendation based on TF-IDF and cosine similarity.

    Input:
    - product_id (int): The ID of the game for which recommendations are requested.

    Output:
    - str: A string representation of a list of recommended game titles.
    """
    id_str = str(product_id)
    game_recomendation_df['features'] = game_recomendation_df['genres'] + ' ' + game_recomendation_df['tags'] + ' ' + game_recomendation_df['specs']
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(game_recomendation_df['features'].fillna(''))
    game_index = game_recomendation_df[game_recomendation_df['id'].astype(str) == id_str].index[0]
    cosine_similarities = linear_kernel(tfidf_matrix[game_index], tfidf_matrix).flatten()
    similar_game_indices = cosine_similarities.argsort()[:-6:-1]
    recommended_titles = game_recomendation_df.iloc[similar_game_indices]['app_name'].tolist()
    return str(recommended_titles)
