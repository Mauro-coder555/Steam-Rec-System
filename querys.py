
import pandas as pd

# Load datasets
items_df = pd.read_csv('data/generated/items.csv', dtype={'item_id': str})
steam_games_df = pd.read_csv('data/generated/steam_games.csv')
reviews_df = pd.read_csv('data/generated/reviews_sentiment.csv', dtype={'item_id': str})



def PlayTimeGenre(genero: str):
    print(items_df.columns)
    print(reviews_df.columns)
    print(items_df['item_id'].dtypes)
    print(reviews_df['item_id'].dtypes)

    # Filtrar juegos por género
    juegos_genero = steam_games_df[steam_games_df['genres'].str.contains(genero, case=False, na=False)]

    # Combinar datasets
    merged_df = pd.merge(items_df, reviews_df, on='item_id')

    # Filtrar por juegos del género especificado
    juegos_genero_ids = juegos_genero['id'].unique()
    juegos_genero_df = merged_df[merged_df['item_id'].isin(juegos_genero_ids)]

    # Agrupar por año y calcular las horas jugadas totales
    horas_por_anio = juegos_genero_df.groupby('release_date')['playtime_forever'].sum()

    # Encontrar el año con más horas jugadas
    año_mas_horas = horas_por_anio.idxmax()

    return {"Año de lanzamiento con más horas jugadas para Género X": int(año_mas_horas)}



def UserForGenre(genero: str):
    # Filtrar juegos por género
    juegos_genero = steam_games_df[steam_games_df['genres'].str.contains(genero, case=False, na=False)]

    # Combinar datasets
    merged_df = pd.merge(items_df, reviews_df, on='item_id')

    # Filtrar por juegos del género especificado
    juegos_genero_ids = juegos_genero['id'].unique()
    juegos_genero_df = merged_df[merged_df['item_id'].isin(juegos_genero_ids)]

    # Agrupar por usuario y año y calcular las horas jugadas totales
    horas_por_usuario_y_anio = juegos_genero_df.groupby(['user_id', 'release_date'])['playtime_forever'].sum()

    # Encontrar el usuario con más horas jugadas
    usuario_mas_horas = horas_por_usuario_y_anio.groupby('user_id').sum().idxmax()

    # Obtener las horas jugadas por año para el usuario encontrado
    horas_por_usuario_y_anio = horas_por_usuario_y_anio.loc[usuario_mas_horas[0]].reset_index()
    horas_por_usuario_y_anio['release_date'] = pd.to_datetime(horas_por_usuario_y_anio['release_date']).dt.year
    horas_por_usuario_y_anio = horas_por_usuario_y_anio.rename(columns={'release_date': 'Año', 'playtime_forever': 'Horas'})

    # Crear la lista de acumulación de horas jugadas por año
    acumulacion_horas_por_anio = horas_por_usuario_y_anio.groupby('Año')['Horas'].sum().reset_index()
    lista_acumulacion_horas = acumulacion_horas_por_anio.to_dict(orient='records')

    return {"Usuario con más horas jugadas para Género X": usuario_mas_horas[0],
            "Horas jugadas": lista_acumulacion_horas}


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