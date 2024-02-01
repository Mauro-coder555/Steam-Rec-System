 
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.responses import RedirectResponse
import importlib
import querys
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

importlib.reload(querys)

app = FastAPI()


# Basic CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/")
def rooth():
    return RedirectResponse(url="/docs/")

@app.get("/developer")
def developer(desarrollador: str):
    result = querys.developer(desarrollador)
    return str(result)

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