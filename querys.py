
import pandas as pd
from datetime import datetime

# Load datasets
items_df = pd.read_csv('data/generated/items.csv', dtype={'item_id': str})
steam_games_df = pd.read_csv('data/generated/steam_games.csv',  dtype={'id': str})
reviews_df = pd.read_csv('data/generated/reviews_sentiment.csv', dtype={'item_id': str})


def prueba(genero: str):
    return "hola"

# Combinar la información relevante de los dataframes
merged_df = pd.merge(items_df, steam_games_df, left_on='item_id', right_on='id')

def PlayTimeGenre(genero):
    # Filtrar por el género especificado y casos donde release date no sea "Not specified"
    genre_df = merged_df[(merged_df['genres'].str.contains(genero, case=False, na=False)) & 
                         (merged_df['release_date'] != "Not specified") &
                         (merged_df['id'] != "Not specified")]

    # Convertir 'release_date' a datetime
    genre_df['release_date'] = pd.to_datetime(genre_df['release_date'], errors='coerce')

    # Eliminar filas con fechas no válidas
    genre_df = genre_df.dropna(subset=['release_date'])

    # Agrupar por año y sumar las horas jugadas
    year_playtime = genre_df.groupby(genre_df['release_date'].dt.year)['playtime_forever'].sum()

    # Encontrar el año con más horas jugadas
    max_year = year_playtime.idxmax()

    return str({f"Año de lanzamiento con más horas jugadas para el género {genero}": max_year})




def parse_date(date_str):
    try:
        return datetime.strptime(date_str, 'Posted %B %d, %Y.')
    except ValueError:
        try:
            # Intentar parsear sin el año
            return datetime.strptime(date_str, 'Posted %B %d.').replace(year=datetime.now().year)
        except ValueError:
            # En caso de error, devolver una fecha con día 1
            return datetime(datetime.now().year, 1, 1)


def UserForGenre(genero: str):
    # Filtrar el DataFrame de steam_games por el género proporcionado
    genre_games = steam_games_df[steam_games_df['genres'].str.contains(genero, case=False, na=False)]

    # Obtener los ids de los juegos del género especificado
    genre_game_ids = genre_games['id'].tolist()

    # Filtrar el DataFrame de items por los juegos del género
    genre_items = items_df[items_df['item_id'].isin(genre_game_ids)]

    # Combinar el DataFrame de items con el DataFrame de reviews
    merged_df = pd.merge(reviews_df, genre_items, on=['item_id', 'user_id'], how='inner')


    # Filtrar las filas donde las horas jugadas no son "Not specified"
    merged_df = merged_df[merged_df['playtime_forever'] != 'Not specified']

    # Convertir las horas jugadas a números enteros
    merged_df['playtime_forever'] = merged_df['playtime_forever'].astype(int)

    # Obtener el usuario con más horas jugadas para el género dado
    max_hours_user = merged_df.groupby('user_id')['playtime_forever'].sum().idxmax()

    # Crear una lista de acumulación de horas jugadas por año
    hours_by_year = []
    for index, row in merged_df.iterrows():
        try:
            posted_date = datetime.strptime(row['posted'], 'Posted %B %d, %Y.')
            year = posted_date.year
        except ValueError:
            # Para fechas sin año especificado
            year = 'Year not specified'

        hours_by_year.append({'Año': year, 'Horas': row['playtime_forever']})

    # Filtrar y sumar las horas jugadas por año
    hours_by_year = pd.DataFrame(hours_by_year).groupby('Año')['Horas'].sum().reset_index()

    # Formatear el resultado como un diccionario
    result = {
        "Usuario con más horas jugadas para Género {}".format(genero): max_hours_user,
        "Horas jugadas": hours_by_year.to_dict(orient='records')
    }

    return str(result)

def UsersRecommend(anio: int):
    # Filtrar las revisiones para el año dado
    reviews_anio = reviews_df[reviews_df['posted'].str.contains(str(anio), na=False)]

    # Filtrar las revisiones recomendadas y positivas/neutrales
    recommended_reviews = reviews_anio[(reviews_anio['recommend'] == True) & (reviews_anio['sentiment_analysis'] > 0)]

    # Combinar con el DataFrame de items para obtener información sobre los juegos
    merged_df = pd.merge(recommended_reviews, items_df, on='item_id', how='inner')

    # Contar las recomendaciones para cada juego
    recommendations_count = merged_df.groupby('item_name')['recommend'].count().reset_index()

    # Obtener el top 3 de juegos más recomendados
    top3_games = recommendations_count.nlargest(3, 'recommend')

    # Formatear el resultado como una lista de diccionarios con puestos fijos
    result = [{"Puesto {}".format(puesto): row['item_name']} for puesto, (_, row) in zip([1, 2, 3], top3_games.iterrows())]

    return str(result)




def UsersWorstDeveloper(anio: int):
    # Filtrar las revisiones para el año dado
    reviews_anio = reviews_df[reviews_df['posted'].str.contains(str(anio), na=False)]

    # Filtrar las revisiones no recomendadas y con comentarios negativos (incluyendo el valor "0")
    worst_reviews = reviews_anio[(reviews_anio['recommend'] == False) & (reviews_anio['sentiment_analysis'] <= 0)]

    # Combinar con el DataFrame de items para obtener información sobre los juegos
    merged_df = pd.merge(worst_reviews, items_df, on='item_id', how='inner')

    # Combinar con el DataFrame de steam_games para obtener información sobre los desarrolladores
    merged_df = pd.merge(merged_df, steam_games_df[['id', 'developer']], left_on='item_id', right_on='id', how='inner')

    # Contar las no recomendaciones para cada desarrolladora
    worst_developers_count = merged_df.groupby('developer')['recommend'].count().reset_index()

    # Obtener el top 3 de desarrolladoras con juegos menos recomendados
    top3_worst_developers = worst_developers_count.nlargest(3, 'recommend')

    # Formatear el resultado como una lista de diccionarios
    result = [{"Puesto {}".format(i + 1): row[1]['developer']} for i, row in enumerate(top3_worst_developers.sort_values('developer').iterrows(), start=0)]

    return str(result)


def sentiment_analysis(empresa_desarrolladora: str):
    # Filtrar las reseñas para la empresa desarrolladora dada
    filtered_reviews = reviews_df[reviews_df['item_id'].isin(
        steam_games_df[steam_games_df['developer'] == empresa_desarrolladora]['id'])]

    # Filtrar las reseñas con valor 'Not specified' en la columna 'sentiment_analysis'
    filtered_reviews = filtered_reviews[filtered_reviews['sentiment_analysis'] != 'Not specified']

    # Contar la cantidad de registros por categoría de análisis de sentimiento
    sentiment_counts = filtered_reviews['sentiment_analysis'].astype(int).value_counts()

    # Crear el diccionario de retorno
    result_dict = {empresa_desarrolladora: {
        'Negative': sentiment_counts.get(0, 0),
        'Neutral': sentiment_counts.get(1, 0),
        'Positive': sentiment_counts.get(2, 0)
    }}

    return str(result_dict)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def recomendacion_juego(product_id):
   # Paso 1: Preprocesamiento y combinación de datos
    merged_df = pd.merge(steam_games_df, reviews_df, left_on='id', right_on='item_id')

    # Paso 2: Feature Engineering - Utiliza TF-IDF para el texto relevante
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(
        merged_df['tags'] + ' ' + merged_df['genres'] + ' ' + merged_df['sentiment_analysis'].astype(str)
    )

    # Paso 3: Modelo de Similitud - Calcula la similitud del coseno
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Paso 4: Generación de Recomendaciones
    game_index = merged_df[merged_df['id'] == product_id].index[0]
    sim_scores = list(enumerate(cosine_sim[game_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    similar_games_indices = [i[0] for i in sim_scores]

    # Paso 5: Retorno de Resultados
    recommended_games = merged_df.iloc[similar_games_indices]['title'].tolist()
    return str(recommended_games)