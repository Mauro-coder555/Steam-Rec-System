 
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.responses import RedirectResponse
import importlib
import querys
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

importlib.reload(querys)

app = FastAPI()


# Configuración básica de CORS
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


@app.get("/Prueba")
async def PlayTimeGenre(genero: str):  
    return querys.prueba(str)


@app.get("/PlayTimeGenre")
async def PlayTimeGenre(genero: str):
    result = querys.PlayTimeGenre(genero)
    return str(result)

@app.get("/UserForGenre")
def UserForGenre(genero: str):
    result = querys.UserForGenre(genero)
    return str(result)

@app.get("/UsersRecommend")
def UsersRecommend(año: str):
    result = querys.UsersRecommend(año)
    return str(result)

@app.get("/UsersWorstDeveloper")
def UsersWorstDeveloper(año: str):
    result = querys.UsersWorstDeveloper(año)
    return str(result)

@app.get("/sentiment_analysis")
def sentiment_analysis(empresa_desarrolladora: str):
    result = querys.sentiment_analysis(empresa_desarrolladora)
    return str(result)