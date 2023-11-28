
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel






# Load datasets
main_df = pd.read_csv('simplified-data/main_dataframe.csv', dtype={'item_id': str, 'user_id': str})

def prueba(genero: str):
    return "hola"

def PlayTimeGenre(genero):
    genre_df = main_df[(main_df['genres'].str.contains(genero, case=False, na=False)) & 
                       (main_df['release_date'] != "Not specified") &
                       (main_df['id'] != "Not specified")]

    genre_df['release_date'] = pd.to_datetime(genre_df['release_date'], errors='coerce')

    genre_df = genre_df.dropna(subset=['release_date'])

    year_playtime = genre_df.groupby(genre_df['release_date'].dt.year)['playtime_forever'].sum()

    max_year = year_playtime.idxmax()

    return str({f"Año de lanzamiento con más horas jugadas para el género {genero}": max_year})

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, 'Posted %B %d, %Y.')
    except ValueError:
        try:
            return datetime.strptime(date_str, 'Posted %B %d.').replace(year=datetime.now().year)
        except ValueError:
            return datetime(datetime.now().year, 1, 1)

def UserForGenre(genero: str):
    genre_games = main_df[main_df['genres'].str.contains(genero, case=False, na=False)]
    genre_game_ids = genre_games['item_id'].tolist()

    genre_items = main_df[main_df['item_id'].isin(genre_game_ids)]

    merged_df = genre_items.copy()

    merged_df = merged_df[merged_df['playtime_forever'] != 'Not specified']
    merged_df['playtime_forever'] = merged_df['playtime_forever'].astype(int)

    max_hours_user = merged_df.groupby('user_id')['playtime_forever'].sum().idxmax()

    hours_by_year = []
    for _, row in merged_df.iterrows():
        try:
            posted_date = datetime.strptime(row['posted'], 'Posted %B %d, %Y.')
            year = posted_date.year
        except ValueError:
            year = 'Year not specified'

        hours_by_year.append({'Año': year, 'Horas': row['playtime_forever']})

    hours_by_year = pd.DataFrame(hours_by_year).groupby('Año')['Horas'].sum().reset_index()

    result = {
        "Usuario con más horas jugadas para Género {}".format(genero): max_hours_user,
        "Horas jugadas": hours_by_year.to_dict(orient='records')
    }

    return str(result)

def UsersRecommend(anio: int):
    reviews_anio = main_df[main_df['posted'].str.contains(str(anio), na=False)]

    recommended_reviews = reviews_anio[(reviews_anio['recommend'] == True) & (reviews_anio['sentiment_analysis'] > 0)]

    merged_df = recommended_reviews.copy()

    recommendations_count = merged_df.groupby('title')['recommend'].count().reset_index()

    top3_games = recommendations_count.nlargest(3, 'recommend')

    result = [{"Puesto {}".format(puesto): row['title']} for puesto, (_, row) in zip([1, 2, 3], top3_games.iterrows())]

    return str(result)

def UsersWorstDeveloper(anio: int):
    reviews_anio = main_df[main_df['posted'].str.contains(str(anio), na=False)]

    worst_reviews = reviews_anio[(reviews_anio['recommend'] == False) & (reviews_anio['sentiment_analysis'] <= 0)]

    merged_df = worst_reviews.copy()

    merged_df = pd.merge(merged_df, main_df[['item_id', 'developer']], on='item_id', how='inner')

    worst_developers_count = merged_df.groupby('developer')['recommend'].count().reset_index()

    top3_worst_developers = worst_developers_count.nlargest(3, 'recommend')

    result = [{"Puesto {}".format(i + 1): row[1]['developer']} for i, row in enumerate(top3_worst_developers.sort_values('developer').iterrows(), start=0)]

    return str(result)

def sentiment_analysis(empresa_desarrolladora: str):
    filtered_reviews = main_df[main_df['item_id'].isin(
        main_df[main_df['developer'] == empresa_desarrolladora]['item_id'])]

    filtered_reviews = filtered_reviews[filtered_reviews['sentiment_analysis'] != 'Not specified']

    sentiment_counts = filtered_reviews['sentiment_analysis'].astype(int).value_counts()

    result_dict = {empresa_desarrolladora: {
        'Negative': sentiment_counts.get(0, 0),
        'Neutral': sentiment_counts.get(1, 0),
        'Positive': sentiment_counts.get(2, 0)
    }}

    return str(result_dict)

def recomendacion_juego(product_id):
    main_df['features'] = main_df['genres'] + ' ' + main_df['tags'] + ' ' + main_df['specs']

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(main_df['features'].fillna(''))

    game_index = main_df[main_df['id'] == product_id].index[0]

    cosine_similarities = linear_kernel(tfidf_matrix[game_index], tfidf_matrix).flatten()

    similar_game_indices = cosine_similarities.argsort()[:-6:-1]

    recommended_titles = main_df.iloc[similar_game_indices]['title'].tolist()

    return str(recommended_titles)