
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from itertools import islice



# Load datasets


developer_df = pd.read_csv('simplified-data/developer_df.csv')
userdata_df = pd.read_csv('simplified-data/userdata_df.csv')
user_for_genre_df = pd.read_csv('simplified-data/user_for_genre.csv')
best_developer_year_df = pd.read_csv('simplified-data/best_developer_year.csv')
developer_reviews_analysis_df = pd.read_csv('simplified-data/sentiment_analysis.csv')
game_recomendation_df = pd.read_csv('simplified-data/game_recomendation.csv')



def developer(desarrollador: str) -> dict:
    """
    Retrieve information for a given developer.

    Parameters:
    - developer_name (str): The name of the developer to retrieve information for.

    Returns:
    - dict: A dictionary containing information about the developer, including 'year',
            'items quantity', 'free content', and 'free content percentage'.
            If the developer is not found, the dictionary will contain a message.
    """
    # Filter the dataframe by the provided developer
    desarrollador = desarrollador.lower()
    developer_data = developer_df[developer_df['developer'] == desarrollador]

    # Check if the developer was found
    if developer_data.empty:
        return {"message": f"No data found for developer: {desarrollador}"}
    else:
        # Retrieve 'year', 'items quantity', and 'free content' columns
        year = developer_data['year'].values[0]
        items_quantity = developer_data['items_quantity'].values[0]
        free_content_percentage = developer_data['free_content_percentage'].values[0]

        
        # Return a dictionary with the requested columns and percentage
        return {
            'Año': year,
            'Cantidad de items': items_quantity,
            'Contenido Free': free_content_percentage
        }

def userdata(User_id: str) -> dict:
    """
    Retrieve information for a given user.

    Parameters:
    - user_id (str): The ID of the user to retrieve information for.

    Returns:
    - dict: A dictionary containing information about the user, including
            'spent money', 'recommendation percentage', and 'items quantity'.
            If the user is not found, the dictionary will contain a message.
    """
    # Check if the user exists in the user_info_df dataframe
    user_data = userdata_df[userdata_df['user_id'] == User_id]


    #The structure of the user IDs must be exactly the same as the one entered,
    #since the variation of lowercase/uppercase letters could refer to different users.

    # Check if the user was found
    if user_data.empty:
        return {"message": f"No data found for user: {User_id}"}
    else:
        # Retrieve 'spent money', 'recommendation percentage', and 'items quantity'
        spent_money = user_data['spent_money'].values[0]
        recommendation_percentage = user_data['recommendation percentage'].values[0]
        items_quantity = user_data['items_quantity'].values[0]

        # Return a dictionary with the requested information
        return {
            'spent money': spent_money,
            'recommendation percentage': recommendation_percentage,
            'items quantity': items_quantity
        }


def UserForGenre(genre: str) -> str:
    """
    Filter the DataFrame for the specified genre, find the user with the most played hours, and create a list of accumulated hours played per year.

    Input:
    - genre (str): The genre for filtering the DataFrame.
      Uppercase or lowercase of the parameter will not affect the result

    Output:
    - str: A string representation of a dictionary containing the user with the most played hours for the specified genre and a list of accumulated hours per year.
    """
    genre = genre.lower()
    df_genre = user_for_genre_df[user_for_genre_df['genres'] == genre]
    user_most_hours = df_genre.loc[df_genre['playtime_forever'].idxmax(), 'user']
    accumulation_hours_per_year = df_genre.groupby('year')['playtime_forever'].sum().reset_index()
    accumulation_hours_per_year = accumulation_hours_per_year.rename(columns={'year': 'Año', 'playtime_forever': 'Horas'})
    list_accumulation_hours = accumulation_hours_per_year.to_dict(orient='records')
    to_return = {
        "Usuario con más horas jugadas para Género {}".format(genre): user_most_hours,
        "Horas jugadas": list_accumulation_hours
    }
    return str(to_return)



def best_developer_year(year: int) -> str:
    """
    Filter the DataFrame for the provided year, group by developer, sum the most recommended count,
    and create a string representation of a top-3 list of dictionaries containing the developers and their total recommended counts.

    Input:
    - year (int): The year for filtering the DataFrame.

    Output:
    - str: A string representation of a top-3 list of dictionaries containing the developers and their total recommended counts.
      If the provided year is not within the valid range in the dataset, returns "Invalid year".
    """
    # Convert the input year to an integer
    year = int(year)
    
    # Check if the provided year is within the valid range in the dataset
    valid_year_range = (best_developer_year_df['year'].min(), best_developer_year_df['year'].max())
    if valid_year_range[0] <= year <= valid_year_range[1]:
        # Filter the DataFrame for the specified year
        df_filtered = best_developer_year_df[best_developer_year_df['year'] == year]
        
        # Group by developer and sum the most recommended count
        df_grouped = df_filtered.groupby('developer')['most_recommended_count'].sum().reset_index()
        
        # Sort the DataFrame by the total recommended count in descending order
        df_sorted = df_grouped.sort_values(by='most_recommended_count', ascending=False)
        
        # Create a string representation of a top-3 list of dictionaries with the developers and their total recommended counts
        resultado = ', '.join(["{{'Puesto {}': '{}'}}".format(i + 1, row['developer']) for i, (_, row) in enumerate(islice(df_sorted.iterrows(), 3))])
        return "[" + resultado + "]"
    else:
        return "Invalid year"

def developer_reviews_analysis(developer_company: str) -> dict:
    """
    Filter the DataFrame for the specified developer, perform sentiment analysis, and return a dictionary of sentiment counts.

    Input:
    - developer_company (str): The developer's name for filtering the DataFrame.
      Uppercase or lowercase of the parameter will not affect the result
    Output:
    - dict: A dictionary containing sentiment counts for the specified developer.
    """
    developer_company = developer_company.lower()
    company_df = developer_reviews_analysis_df[developer_reviews_analysis_df['developer'] == developer_company]
    result_dict = {developer_company: []}
    sentiment_columns = ['Negative', 'Neutral', 'Positive']
    for sentiment in sentiment_columns:
        count = company_df[sentiment].values[0]
        result_dict[developer_company].append({sentiment: count})
    return result_dict


def recomendacion_juego(product_input: str) -> list:
    """
    Recommends similar games based on content using TF-IDF and cosine similarity.

    Input:
    - product_input (str): Either the ID or the name of the game.
      Uppercase or lowercase of the parameter will not affect the result
    Output:
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
