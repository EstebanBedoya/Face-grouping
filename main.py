import uvicorn
from app.views import app

if __name__ == "__main__":
    # Configuración para ejecutar el servidor FastAPI localmente
    # Ejecuta con: python main.py
    uvicorn.run(
        "app.views:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # Auto-reload en desarrollo
        log_level="info"
    )

# También funciona con: uvicorn main:app --reload 
