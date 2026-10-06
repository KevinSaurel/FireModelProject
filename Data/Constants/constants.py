from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Ruta base funcional de Open Data Euskadi
API_URL = "https://api.euskadi.eus/euskalmet/stations"

# Claims obligatorios para el token JWT
JWT_AUD = "met01.apikey"
JWT_ISS = "FireModelProject"
JWT_VERSION = "1.0.0"
USER_EMAIL = "kevin.saurel@opendeusto.es"

DB_PATH = BASE_DIR / "Database" / "incendios_euskadi.db"
PRIVATE_KEY_PATH = BASE_DIR / "private_key.pem"