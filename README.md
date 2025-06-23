# Face Grouping MVP

Agrupa fotos automáticamente por rostro usando FastAPI, DeepFace y una interfaz web moderna.

## 🚀 Características
- Subida de imágenes por web (drag & drop o selección)
- Agrupación automática de rostros usando DeepFace (Facenet + RetinaFace)
- Visualización de grupos con miniaturas y galería
- Descarga de grupos como ZIP
- Elimina imágenes individuales antes de procesar
- 100% local, sin necesidad de consola para el usuario final

## 📦 Instalación

```bash
# 1. Clona el repositorio
 git clone https://github.com/tuusuario/group-by-faces.git
 cd group-by-faces

# 2. Crea un entorno virtual e instala dependencias
python -m venv env
source env/bin/activate  # En Windows: env\Scripts\activate
pip install -r requirements.txt

# 3. Ejecuta la app
python main.py
# o
uvicorn main:app --reload
```

Abre tu navegador en [http://localhost:8000](http://localhost:8000)

## 🗂️ Estructura del proyecto

```
project-root/
│
├── app/
│   ├── templates/
│   │   └── index.html         # Interfaz web
│   ├── static/               # (opcional) Estilos extra
│   ├── views.py              # Rutas FastAPI
│   └── grouping.py           # Lógica DeepFace
│
├── input_photos/             # Imágenes subidas (ignorada en git)
├── grouped_photos/           # Grupos generados (ignorada en git)
├── main.py                   # Lanza el servidor
├── requirements.txt
├── .gitignore
└── README.md
```

## 📝 Licencia
MIT

---

**Desarrollado con ❤️ usando FastAPI, DeepFace y Tailwind CSS.** 