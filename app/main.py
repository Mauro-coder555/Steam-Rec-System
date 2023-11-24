 
from fastapi import FastApi
from fastapi.responses import JSONResponse
from starlette.responses import RedirectResponse
import importlib
import querys


importlib.reload(querys)

app = FastApi()

@app.get("/")
def rooth():
    return RedirectResponse(url="/docs/")





@app.get("/PlayTimeGenre")
async def PlayTimeGenre(genero: str):
    result = querys.PlayTimeGenre(genero)
    return JSONResponse(content=result)

@app.get("/UserForGenre")
async def UserForGenre(genero: str):
    result = querys.UserForGenre(genero)
    return JSONResponse(content=result)

@app.get("/UsersRecommend")
async def UsersRecommend(año: str):
    result = querys.UsersRecommend(año)
    return JSONResponse(content=result)

@app.get("/UsersWorstDeveloper")
async def UsersWorstDeveloper(año: str):
    result = querys.UsersWorstDeveloper(año)
    return JSONResponse(content=result)

@app.get("/sentiment_analysis")
async def sentiment_analysis(empresa_desarrolladora: str):
    result = querys.sentiment_analysis(empresa_desarrolladora)
    return JSONResponse(content=result)