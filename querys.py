
import pandas as pd

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





def UserForGenre(genero):
    # Combinar la información relevante de los dataframes
    merged_df = pd.merge(items_df, reviews_df, on='item_id')

    # Filtrar por el género especificado y casos donde release date y user_id no sean "Not specified"
    genre_df = merged_df[(merged_df['genres'].str.contains(genero, case=False, na=False)) & 
                         (merged_df['release_date'] != "Not specified") &
                         (merged_df['user_id'] != "Not specified")]

    # Convertir 'release_date' a datetime
    genre_df['release_date'] = pd.to_datetime(genre_df['release_date'], errors='coerce')

    # Eliminar filas con fechas no válidas
    genre_df = genre_df.dropna(subset=['release_date'])

    # Agrupar por usuario y sumar las horas jugadas
    user_playtime = genre_df.groupby('user_id')['playtime_forever'].sum()

    # Encontrar el usuario con más horas jugadas
    max_user = user_playtime.idxmax()

    # Filtrar por el usuario con más horas jugadas
    max_user_df = genre_df[genre_df['user_id'] == max_user]

    # Agrupar por año y sumar las horas jugadas
    year_playtime = max_user_df.groupby(max_user_df['release_date'].dt.year)['playtime_forever'].sum()

    # Crear la lista de acumulación de horas jugadas por año
    horas_por_anio = [{"Año": int(year), "Horas": int(playtime)} for year, playtime in year_playtime.items()]

    # Crear el diccionario de retorno
    result = {
        "Usuario con más horas jugadas para Género": str(max_user),
        "Horas jugadas": horas_por_anio
    }

    return str(result)


def UsersRecommend(anio: str):
    anio = int(anio)

     # Combinar datasets
    merged_df = pd.merge(items_df, reviews_df, on='item_id')

    # Filtrar por año y recomendaciones positivas/neutrales
    juegos_anio_recomendados = merged_df[(merged_df['recommend'] == True) & (pd.to_datetime(merged_df['last_edited']).dt.year == anio)]

    # Contar las recomendaciones por juego
    recomendaciones_por_juego = juegos_anio_recomendados.groupby('item_name')['recommend'].sum()

    # Obtener el top 3 de juegos más recomendados
    top3_juegos_recomendados = recomendaciones_por_juego.nlargest(3).reset_index()

    # Crear el formato de retorno
    lista_top3 = [{"Puesto {}: {}".format(i + 1, juego): recomendaciones} for i, (juego, recomendaciones) in enumerate(top3_juegos_recomendados.values)]

    return lista_top3

def UsersWorstDeveloper(anio: str):
    anio = int(anio)
    
    # Combinar datasets
    merged_df = pd.merge(items_df, reviews_df, on='item_id')

    # Filtrar por año y recomendaciones negativas
    juegos_anio_no_recomendados = merged_df[(merged_df['recommend'] == False) & (pd.to_datetime(merged_df['last_edited']).dt.year == anio)]

    # Obtener el top 3 de desarrolladoras con juegos menos recomendados
    top3_peores_desarrolladoras = juegos_anio_no_recomendados.groupby('developer')['recommend'].count().nlargest(3).reset_index()

    # Crear el formato de retorno
    lista_top3 = [{"Puesto {}: {}".format(i + 1, desarrolladora): juegos_no_recomendados} for i, (desarrolladora, juegos_no_recomendados) in enumerate(top3_peores_desarrolladoras.values)]

    return lista_top3

def sentiment_analysis(desarrolladora: str):
    # Filtrar por la empresa desarrolladora proporcionada
    reviews_desarrolladora = reviews_df[reviews_df['developer'] == desarrolladora]

    # Contar la cantidad total de registros de reseñas por análisis de sentimiento
    conteo_sentimientos = reviews_desarrolladora['sentiment_analysis'].value_counts().to_dict()

    # Crear el formato de retorno
    resultado_sentimiento = {desarrolladora: [{'Negative': conteo_sentimientos.get('Negative', 0)},
                                              {'Neutral': conteo_sentimientos.get('Neutral', 0)},
                                              {'Positive': conteo_sentimientos.get('Positive', 0)}]}

    return resultado_sentimiento