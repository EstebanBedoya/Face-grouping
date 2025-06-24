from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
import json
import zipfile
from typing import List
from pathlib import Path
import aiofiles
from .grouping import group_faces

# Initialize FastAPI app
app = FastAPI(title="Face Grouping MVP", description="Agrupa fotos por rostro automáticamente")

# Mount static files
# app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")

# Ensure directories exist
INPUT_FOLDER = "input_photos"
OUTPUT_FOLDER = "grouped_photos"

os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Página principal con formulario de carga y visualización de resultados.
    Muestra grupos existentes si los hay.
    """
    # Verificar si existen resultados previos
    groups = []
    processing_info = None
    
    if os.path.exists("processing_results.json"):
        try:
            with open("processing_results.json", 'r') as f:
                processing_info = json.load(f)
                groups = processing_info.get('groups', [])
        except Exception as e:
            print(f"Error loading previous results: {e}")
    
    # Obtener imágenes en input_folder
    input_images = []
    if os.path.exists(INPUT_FOLDER):
        for filename in os.listdir(INPUT_FOLDER):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                input_images.append({
                    'filename': filename,
                    'path': f"/input_image/{filename}"
                })
    
    input_count = len(input_images)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "groups": groups,
        "input_images": input_images,
        "input_count": input_count,
        "processing_info": processing_info
    })

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Sube múltiples archivos de imagen al directorio input_photos/.
    """
    uploaded_files = []
    errors = []
    
    for file in files:
        if file.filename is None:
            errors.append("File has no filename")
            continue
            
        if file.content_type and file.content_type.startswith('image/'):
            # Validate file extension
            file_extension = Path(file.filename).suffix.lower()
            if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                try:
                    # Save file to input folder
                    file_path = os.path.join(INPUT_FOLDER, file.filename)
                    async with aiofiles.open(file_path, 'wb') as f:
                        content = await file.read()
                        await f.write(content)
                    uploaded_files.append(file.filename)
                except Exception as e:
                    errors.append(f"Error uploading {file.filename}: {str(e)}")
            else:
                errors.append(f"Unsupported file format: {file.filename}")
        else:
            errors.append(f"Invalid file type: {file.filename}")
    
    return {
        "uploaded": uploaded_files,
        "errors": errors,
        "total_uploaded": len(uploaded_files)
    }

@app.post("/process")
async def process_images():
    """
    Procesa las imágenes subidas y las agrupa por rostros usando DeepFace.
    """
    try:
        # Verificar que hay imágenes para procesar
        image_files = [f for f in os.listdir(INPUT_FOLDER) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
        
        if not image_files:
            raise HTTPException(status_code=400, detail="No images found in input folder")
        
        # Ejecutar el procesamiento de agrupación
        results = group_faces(INPUT_FOLDER, OUTPUT_FOLDER)
        
        if results['success']:
            return {
                "success": True,
                "message": results['message'],
                "groups": results['groups'],
                "stats": results['stats']
            }
        else:
            raise HTTPException(status_code=500, detail=results['message'])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/results")
async def get_results():
    """
    Obtiene los resultados del procesamiento y información de grupos.
    """
    try:
        # Check if processing results exist
        if os.path.exists("processing_results.json"):
            with open("processing_results.json", 'r') as f:
                results = json.load(f)
        else:
            results = None
        
        # Get group folders info
        groups = []
        if os.path.exists(OUTPUT_FOLDER):
            for group_folder in os.listdir(OUTPUT_FOLDER):
                group_path = os.path.join(OUTPUT_FOLDER, group_folder)
                if os.path.isdir(group_path):
                    images = [f for f in os.listdir(group_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
                    groups.append({
                        "name": group_folder,
                        "image_count": len(images),
                        "images": images
                    })
        
        return {
            "processing_results": results,
            "groups": groups
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting results: {str(e)}")

@app.get("/image/{group_name}/{filename}")
async def get_image(group_name: str, filename: str):
    """
    Sirve una imagen específica de un grupo para mostrar en la galería.
    """
    image_path = os.path.join(OUTPUT_FOLDER, group_name, filename)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(image_path)

@app.get("/download/{group_name}")
async def download_group(group_name: str):
    """
    Descarga un grupo específico como archivo ZIP.
    """
    group_path = os.path.join(OUTPUT_FOLDER, group_name)
    
    if not os.path.exists(group_path):
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Crear archivo ZIP temporal
    zip_filename = f"{group_name}.zip"
    zip_path = os.path.join(OUTPUT_FOLDER, zip_filename)
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename in os.listdir(group_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                    file_path = os.path.join(group_path, filename)
                    zipf.write(file_path, filename)
        
        return FileResponse(
            zip_path, 
            media_type='application/zip',
            filename=zip_filename,
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating ZIP file: {str(e)}")

@app.delete("/clear")
async def clear_all():
    """
    Limpia todas las imágenes subidas y resultados de procesamiento.
    """
    try:
        # Clear input folder
        if os.path.exists(INPUT_FOLDER):
            for file in os.listdir(INPUT_FOLDER):
                file_path = os.path.join(INPUT_FOLDER, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        # Clear output folder
        if os.path.exists(OUTPUT_FOLDER):
            shutil.rmtree(OUTPUT_FOLDER)
            os.makedirs(OUTPUT_FOLDER)
        
        # Remove processing results
        if os.path.exists("processing_results.json"):
            os.remove("processing_results.json")
        
        return {"success": True, "message": "All data cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing data: {str(e)}")

@app.get("/status")
async def get_status():
    """
    Obtiene el estado actual de la aplicación.
    """
    input_count = len([f for f in os.listdir(INPUT_FOLDER) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))])
    
    group_count = 0
    total_grouped_images = 0
    
    if os.path.exists(OUTPUT_FOLDER):
        for group_folder in os.listdir(OUTPUT_FOLDER):
            group_path = os.path.join(OUTPUT_FOLDER, group_folder)
            if os.path.isdir(group_path):
                group_count += 1
                images = [f for f in os.listdir(group_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
                total_grouped_images += len(images)
    
    return {
        "input_images": input_count,
        "groups_created": group_count,
        "grouped_images": total_grouped_images,
        "has_results": os.path.exists("processing_results.json")
    }

@app.get("/input_image/{filename}")
async def get_input_image(filename: str):
    """
    Sirve una imagen específica de la carpeta input_photos para mostrar en la galería.
    """
    image_path = os.path.join(INPUT_FOLDER, filename)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(image_path)

@app.get("/input_images")
async def get_input_images():
    """
    Obtiene la lista de imágenes en la carpeta input_photos.
    """
    input_images = []
    if os.path.exists(INPUT_FOLDER):
        for filename in os.listdir(INPUT_FOLDER):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                input_images.append({
                    'filename': filename,
                    'path': f"/input_image/{filename}"
                })
    
    return {
        "images": input_images,
        "count": len(input_images)
    }

@app.delete("/input_image/{filename}")
async def delete_input_image(filename: str):
    """
    Elimina una imagen específica de la carpeta input_photos.
    """
    image_path = os.path.join(INPUT_FOLDER, filename)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        os.remove(image_path)
        return {"success": True, "message": f"Image {filename} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting image: {str(e)}") 