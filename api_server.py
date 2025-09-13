from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import uvicorn
import time

# Ajouter le chemin vers BallonsTranslator
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import simple pour commencer
translator_ready = False

try:
    # Test import basique
    import modules
    print("✅ Modules BallonsTranslator détectés")
    translator_ready = True
except ImportError as e:
    print(f"⚠️ Import BallonsTranslator échoué: {e}")
    print("API fonctionnera en mode simulation pour l'instant")

app = FastAPI(
    title="Balloons Translator API",
    description="API REST pour traduction d'images manga en temps réel",
    version="1.0.0"
)

# CORS pour permettre les requêtes depuis l'extension Chrome
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles de données
class TranslationRequest(BaseModel):
    image_base64: str
    source_lang: str = "ja"
    target_lang: str = "en"
    translator: str = "google"

class TranslationResponse(BaseModel):
    success: bool
    translated_image_base64: str = None
    error: str = None
    processing_time: float = 0

# Variables globales (simplifiées pour l'instant)
translator_ready = False

@app.on_event("startup")
async def startup_event():
    """Initialisation des modules au démarrage"""
    global translator_ready
    
    try:
        print("🔄 Initialisation de l'API...")
        print("📁 Dossier de travail:", os.getcwd())
        
        # Pour l'instant, mode simulation activé
        translator_ready = True
        print("✅ API prête en mode simulation")
        
    except Exception as e:
        print(f"❌ Erreur initialisation: {e}")
        translator_ready = False

@app.get("/")
async def root():
    """Point d'entrée de base"""
    return {
        "message": "Balloons Translator API",
        "status": "running",
        "mode": "simulation" if not translator_ready else "production",
        "endpoints": {
            "translate": "/translate (POST)",
            "health": "/health (GET)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    return {
        "status": "healthy" if translator_ready else "simulation_mode",
        "modules_loaded": translator_ready,
        "working_directory": os.getcwd()
    }

@app.post("/translate", response_model=TranslationResponse)
async def translate_image(request: TranslationRequest):
    """
    Traduire une image manga
    
    Args:
        request: Données de la requête (image base64, langues, traducteur)
        
    Returns:
        TranslationResponse avec l'image traduite ou erreur
    """
    start_time = time.time()
    
    try:
        # Décoder l'image base64
        try:
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Convertir en RGB si nécessaire
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image invalide: {str(e)}")
        
        # TODO: Intégrer la logique de traduction de BallonsTranslator
        # Pour l'instant, mode simulation - ajouter du texte "TRANSLATED" sur l'image
        
        # Simulation de traitement
        print("🔍 [SIMULATION] Détection de texte...")
        time.sleep(0.5)  # Simule le traitement
        
        print("📖 [SIMULATION] Reconnaissance de texte...")
        time.sleep(0.3)
        
        print("🌐 [SIMULATION] Traduction...")
        time.sleep(0.2)
        
        print("🎨 [SIMULATION] Rendu final...")
        time.sleep(0.2)
        
        # Ajouter un texte "TRANSLATED" sur l'image pour tester
        draw = ImageDraw.Draw(image)
        try:
            # Essayer d'utiliser une police par défaut
            font = ImageFont.load_default()
        except:
            font = None
            
        # Ajouter du texte pour montrer que ça marche
        draw.text((10, 10), "TRANSLATED BY API", fill="red", font=font)
        draw.text((10, 30), f"Lang: {request.source_lang}->{request.target_lang}", fill="red", font=font)
        draw.text((10, 50), f"Size: {image.width}x{image.height}", fill="red", font=font)
        
        print(f"Image processed: {image.width}x{image.height}, mode: {image.mode}")
        
        # Simulation: retourner l'image modifiée avec texte "TRANSLATED"
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        result_base64 = base64.b64encode(image_bytes).decode()
        
        print(f"Generated base64 length: {len(result_base64)} characters")
        print(f"Base64 preview: {result_base64[:50]}...")
        
        processing_time = time.time() - start_time
        
        return TranslationResponse(
            success=True,
            translated_image_base64=result_base64,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        return TranslationResponse(
            success=False,
            error=str(e),
            processing_time=processing_time
        )

@app.post("/translate-file")
async def translate_file(file: UploadFile = File(...)):
    """
    Alternative: Upload direct d'un fichier image
    """
    try:
        # Lire le fichier
        image_data = await file.read()
        image_base64 = base64.b64encode(image_data).decode()
        
        # Utiliser l'endpoint principal
        request = TranslationRequest(image_base64=image_base64)
        return await translate_image(request)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    print("🚀 Démarrage Balloons Translator API Server...")
    print("📖 Documentation: http://localhost:8000/docs")
    print("🔧 Health check: http://localhost:8000/health")
    
    uvicorn.run(
        "api_server:app",
        host="127.0.0.1", 
        port=8000, 
        reload=True,  # Rechargement auto en développement
        log_level="info"
    )
