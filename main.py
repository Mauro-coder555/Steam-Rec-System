 
from fastapi import FastAPI, Response
from starlette.responses import RedirectResponse
import importlib
import querys
importlib.reload(querys)

app = FastAPI()




@app.get("/")
def rooth():
    return RedirectResponse(url="/docs/")

@app.get("/developer")
def developer(desarrollador: str):
    result = querys.developer(desarrollador)
    table_string = result.to_string()
    return Response(content=table_string, media_type="text/plain")

@app.get("/userdata")
def userdata(User_id: str):
    result = querys.userdata(User_id)
    return str(result)

@app.get("/UserForGenre")
def UserForGenre(genero: str):
    result = querys.UserForGenre(genero)
    return str(result)

@app.get("/best_developer_year")
def best_developer_year(año: int):
    result = querys.best_developer_year(año)
    return str(result)

@app.get("/developer_reviews_analysis")
def developer_reviews_analysis(desarrolladora: str):
    result = querys.developer_reviews_analysis(desarrolladora)
    return str(result)

@app.get("/game recommendation")
def sentiment_analysis(item: str):
    result = querys.recomendacion_juego(item)
    return str(result)