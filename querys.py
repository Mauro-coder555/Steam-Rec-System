
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from itertools import islice



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
    genre = genre.lower()
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
    genre = genre.lower()
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
    - str: A string representation of a top-3 list of dictionaries containing the game rankings for the specified year.
    """
    year = int(year) if users_recommend_df['year'].min() <= int(year) <= users_recommend_df['year'].max() else "Invalid year"
    if(year == "Invalid year" ):
        return str(year)
    else:
        año_int = int(year)
        df_filtered = users_recommend_df[users_recommend_df['year'].astype(int) == año_int]
        df_sorted = df_filtered.sort_values(by='recommendations_count', ascending=False)
        resultado = [{"Puesto {}".format(i + 1): row['item_id']} for i, (_, row) in enumerate(islice(df_sorted.iterrows(), 3))]
        return str(resultado)


def UsersWorstDeveloper(year: int) -> str:
    """
    Filter the DataFrame for the provided year, sort it by the least recommended count in ascending order, and create a list of dictionaries with the game rankings.

    Input:
    - año (int): The year for filtering the DataFrame.

    Output:
    - str: A string representation of a top-3 list of dictionaries containing the game rankings for the specified year.
    """
    year = int(year) if users_worst_developer_df['year'].min() <= int(year) <= users_worst_developer_df['year'].max() else "Invalid year"
    if(year == "Invalid year" ):
        return str(year)
    else:      
        df_filtered = users_worst_developer_df[users_worst_developer_df['year'].astype(int) == year]
        df_sorted = df_filtered.sort_values(by='least_recommended_count')
        resultado = [{"Puesto {}".format(i + 1): row['item_id']} for i, (_, row) in enumerate(islice(df_sorted.iterrows(), 3))]
        return str(resultado)

def sentiment_analysis(developer_company: str) -> dict:
    """
    Filter the DataFrame for the specified developer, perform sentiment analysis, and return a dictionary of sentiment counts.

    Input:
    - developer_company (str): The developer's name for filtering the DataFrame.

    Output:
    - dict: A dictionary containing sentiment counts for the specified developer.
    """
    developer_company = developer_company.lower()
    company_df = sentiment_analysis_df[sentiment_analysis_df['developer'] == developer_company]
    result_dict = {developer_company: []}
    sentiment_columns = ['Negative', 'Neutral', 'Positive']
    for sentiment in sentiment_columns:
        count = company_df[sentiment].values[0]
        result_dict[developer_company].append({sentiment: count})
    return result_dict


def recomendacion_juego(product_input: str) -> list:
    """
    Recommends similar games based on content using TF-IDF and cosine similarity.

    Args:
        product_input (str): Either the ID or the name of the game.

    Returns:
        list: List containing information about the recommended games (ID and name).
              If no game is found, a message is included in the list.
    """
    product_input = product_input.lower()
    # Check if the input is an ID or a name
    if product_input.isdigit():
        # If it's an ID, convert it to a string
        id_str = str(product_input)
        # Find the index of the game in the DataFrame
        game_index = game_recomendation_df[game_recomendation_df['id'].astype(str) == id_str].index
    else:
        # If it's a name, find the index of the game by name
        game_index = game_recomendation_df[game_recomendation_df['app_name'] == product_input].index

    # Check if a game was found with the provided input
    if len(game_index) == 0:
        return ["No game found with the provided input."]

    # Get the first index (in case there are multiple games with the same name)
    game_index = game_index[0]

    # Combine the 'genres', 'tags', and 'specs' columns into a new 'features' column
    game_recomendation_df['features'] = game_recomendation_df['genres'] + ' ' + game_recomendation_df['tags'] + ' ' + game_recomendation_df['specs']

    # Use TfidfVectorizer to convert the text in 'features' to a TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(game_recomendation_df['features'].fillna(''))

    # Calculate cosine similarities between the selected game and all other games
    cosine_similarities = linear_kernel(tfidf_matrix[game_index], tfidf_matrix).flatten()

    # Get the indices of the most similar games (excluding the input game)
    similar_game_indices = cosine_similarities.argsort()[:-7:-1]
    similar_game_indices = [idx for idx in similar_game_indices if idx != game_index]

    # Get information about the ID and name of the recommended games
    recommended_games_info = game_recomendation_df.iloc[similar_game_indices][['id', 'app_name']].values.tolist()

    return recommended_games_info
