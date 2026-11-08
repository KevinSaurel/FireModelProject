# Usa la imagen base liviana de Python 3.13
FROM python:3.13-slim

WORKDIR /app

# Instalar dependencias del sistema (incluyendo curl para descargar el modelo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar librerías de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del proyecto
COPY . .

# Descargar el modelo binario real de 289 MB desde GitHub Releases
# El parámetro -fL asegura que falle si la URL devuelve un error HTTP (ej. 404 o 403)
# Descargar el modelo binario real de 289 MB desde GitHub Releases con reintentos automáticos
RUN curl -fL --retry 5 --retry-delay 5 --retry-all-errors -o Model/fireTrainModel.pkl "https://github.com/KevinSaurel/FireModelProject/releases/download/v1.0.0/fireTrainModel.pkl"# Exponer el puerto
EXPOSE 8000

# Iniciar Uvicorn leyendo la variable PORT dinámica de Cloud Run
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]