FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

WORKDIR /app

COPY ./app /app

# Instala las dependencias de tu aplicación
RUN pip install --no-cache-dir -r requirements.txt


# Comando para ejecutar tu aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]