import os
import shutil
import json
import numpy as np
from deepface import DeepFace
from typing import List, Dict, Tuple, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    Returns similarity between 0 and 1 (1 = identical, 0 = completely different)
    """
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    return max(0, min(1, similarity))

def get_face_embedding(image_path: str) -> Optional[np.ndarray]:
    """
    Extract face embedding from image using DeepFace with fallback options.
    """
    try:
        # Intenta con RetinaFace primero (más preciso)
        result = DeepFace.represent(
            img_path=image_path,
            model_name='Facenet',
            detector_backend='retinaface',
            enforce_detection=True,
            align=True,
            normalization='Facenet'
        )
        
        if result and len(result) > 0:
            embedding = np.array(result[0]['embedding'])
            normalized_embedding = embedding / np.linalg.norm(embedding)
            return normalized_embedding
            
    except Exception as e:
        logger.warning(f"RetinaFace failed for {image_path}, trying OpenCV: {str(e)}")
        
        # Fallback a OpenCV (más permisivo)
        try:
            result = DeepFace.represent(
                img_path=image_path,
                model_name='Facenet',
                detector_backend='opencv',
                enforce_detection=False,
                align=True,
                normalization='Facenet'
            )
            
            if result and len(result) > 0:
                embedding = np.array(result[0]['embedding'])
                normalized_embedding = embedding / np.linalg.norm(embedding)
                logger.info(f"Successfully processed {image_path} with OpenCV fallback")
                return normalized_embedding
                
        except Exception as e2:
            logger.error(f"Failed to process {image_path}: {str(e2)}")
            
    return None

def find_best_group(embedding: np.ndarray, groups: List[Dict], threshold: float = 0.6) -> Optional[int]:
    """
    Find the best matching group for a face embedding.
    """
    best_group = None
    best_similarity = 0.0
    
    for i, group in enumerate(groups):
        # Calcula similitud con el centroide del grupo
        centroid = np.array(group['centroid'])
        similarity = cosine_similarity(embedding, centroid)
        
        if similarity > threshold and similarity > best_similarity:
            best_similarity = similarity
            best_group = i
    
    return best_group

def update_group_centroid(group: Dict, new_embedding: np.ndarray):
    """
    Update group centroid with new embedding.
    """
    group['embeddings'].append(new_embedding.tolist())
    group['count'] += 1
    
    # Recalcular centroide como promedio de todos los embeddings
    all_embeddings = np.array(group['embeddings'])
    new_centroid = np.mean(all_embeddings, axis=0)
    group['centroid'] = (new_centroid / np.linalg.norm(new_centroid)).tolist()

def group_faces(input_folder: str = "input_photos", output_folder: str = "grouped_photos") -> Dict:
    """
    Main function to group faces from input folder and save to output folder.
    Returns dictionary with results and group information.
    """
    logger.info("🚀 Starting face grouping process...")
    
    # Configuración
    similarity_threshold = 0.6  # Umbral de similitud para agrupar rostros
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    # Limpiar carpeta de resultados previos
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        logger.info(f"✅ Cleared previous results in {output_folder}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Verificar que existe la carpeta de entrada
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder '{input_folder}' not found!")
    
    # Obtener todas las imágenes
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(supported_extensions)]
    
    if not image_files:
        return {
            "success": False,
            "message": "No images found in input folder",
            "groups": [],
            "stats": {"total": 0, "processed": 0, "no_face": 0, "groups_created": 0}
        }
    
    logger.info(f"📁 Found {len(image_files)} images to process")
    
    # Inicializar variables de seguimiento
    groups = []  # Lista de grupos con sus centroides
    processed_files = {}
    stats = {
        "total": len(image_files),
        "processed": 0,
        "no_face": 0,
        "groups_created": 0
    }
    
    # Procesar cada imagen
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(input_folder, image_file)
        logger.info(f"[{idx}/{len(image_files)}] Processing: {image_file}")
        
        # Extraer embedding del rostro
        embedding = get_face_embedding(image_path)
        
        if embedding is None:
            logger.warning(f"❌ No face detected in {image_file}")
            stats["no_face"] += 1
            continue
        
        # Buscar grupo coincidente
        matching_group_idx = find_best_group(embedding, groups, similarity_threshold)
        
        if matching_group_idx is not None:
            # Agregar a grupo existente
            group_name = groups[matching_group_idx]['name']
            logger.info(f"👥 Assigned to existing group: {group_name}")
            update_group_centroid(groups[matching_group_idx], embedding)
        else:
            # Crear nuevo grupo
            group_name = f"person_{len(groups) + 1}"
            logger.info(f"🆕 Created new group: {group_name}")
            
            new_group = {
                'name': group_name,
                'centroid': embedding.tolist(),
                'embeddings': [embedding.tolist()],
                'count': 1,
                'images': []
            }
            groups.append(new_group)
            stats["groups_created"] += 1
            matching_group_idx = len(groups) - 1
        
        # Crear carpeta del grupo y copiar imagen
        group_folder = os.path.join(output_folder, group_name)
        os.makedirs(group_folder, exist_ok=True)
        
        try:
            destination = os.path.join(group_folder, image_file)
            shutil.copy2(image_path, destination)
            groups[matching_group_idx]['images'].append(image_file)
            
            processed_files[image_file] = {
                'group': group_name,
                'path': destination
            }
            stats["processed"] += 1
            logger.info(f"📋 Copied to: {group_name}/{image_file}")
            
        except Exception as e:
            logger.error(f"❌ Error copying {image_file}: {str(e)}")
    
    # Preparar resultados
    result_groups = []
    for group in groups:
        group_info = {
            'name': group['name'],
            'count': group['count'],
            'images': group['images'],
            'folder_path': os.path.join(output_folder, group['name'])
        }
        result_groups.append(group_info)
    
    # Guardar resultados en JSON
    results = {
        'success': True,
        'message': f"Successfully processed {stats['processed']} images into {stats['groups_created']} groups",
        'groups': result_groups,
        'stats': stats,
        'processed_files': processed_files
    }
    
    try:
        with open("processing_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        logger.info("💾 Results saved to processing_results.json")
    except Exception as e:
        logger.error(f"❌ Error saving results: {str(e)}")
    
    # Imprimir resumen
    logger.info("=" * 50)
    logger.info("📊 PROCESSING SUMMARY:")
    logger.info(f"   Total images: {stats['total']}")
    logger.info(f"   Successfully processed: {stats['processed']}")
    logger.info(f"   No face detected: {stats['no_face']}")
    logger.info(f"   Groups created: {stats['groups_created']}")
    logger.info("=" * 50)
    
    return results

if __name__ == "__main__":
    # Para pruebas independientes
    result = group_faces()
    print(f"Process completed: {result['success']}")
    print(f"Message: {result['message']}")
