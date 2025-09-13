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
import numpy as np

# Ajouter le chemin vers BallonsTranslator
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import des modules BallonsTranslator
translator_ready = False
ballons_modules = {}

try:
    # Imports progressifs des modules BallonsTranslator
    print("Loading BallonsTranslator modules...")
    
    # Import des modules de base
    from modules.translators.google import GoogleTranslator
    from modules.textdetector.ctd import ComicTextDetector  
    from modules.ocr.mit48px import Mit48pxOCR
    from modules.inpaint.lama_large_512px import LamaLargeInpainter
    
    print("✅ BallonsTranslator modules imported successfully")
    translator_ready = True
    
except ImportError as e:
    print(f"⚠️ Could not import BallonsTranslator modules: {e}")
    print("Falling back to simulation mode")
    translator_ready = False

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

@app.on_event("startup")
async def startup_event():
    """Initialisation des modules au démarrage"""
    global translator_ready, ballons_modules
    
    try:
        print("🔄 Initialisation de l'API...")
        print("📁 Dossier de travail:", os.getcwd())
        
        if translator_ready:
            print("Loading BallonsTranslator modules...")
            
            # Initialiser les modules
            ballons_modules = {
                'translator': GoogleTranslator(),
                'detector': ComicTextDetector(),
                'ocr': Mit48pxOCR(),
                'inpainter': LamaLargeInpainter()
            }
            
            print("✅ All BallonsTranslator modules loaded successfully")
            
        else:
            print("✅ API prête en mode simulation")
        
    except Exception as e:
        print(f"❌ Erreur initialisation modules: {e}")
        print("Falling back to simulation mode")
        translator_ready = False

@app.get("/")
async def root():
    """Point d'entrée de base"""
    return {
        "message": "Balloons Translator API",
        "status": "running",
        "mode": "production" if translator_ready else "simulation",
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
        
        # Traduction réelle avec BallonsTranslator ou simulation
        if translator_ready and ballons_modules:
            # MODE RÉEL : Utiliser les modules BallonsTranslator
            result_image = await process_with_ballons_translator(image, request)
        else:
            # MODE SIMULATION : Ajouter du texte comme avant
            result_image = process_simulation_mode(image, request)
        
        # Convertir en base64
        buffer = io.BytesIO()
        result_image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        result_base64 = base64.b64encode(image_bytes).decode()
        
        print(f"Generated base64 length: {len(result_base64)} characters")
        
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

# Fonction de traitement réel avec BallonsTranslator
async def process_with_ballons_translator(image, request):
    """Traitement réel avec les modules BallonsTranslator"""
    try:
        print("🔍 [REAL] Détection de texte...")
        
        # Convertir PIL Image en numpy array pour les modules
        img_array = np.array(image)
        
        # 1. Détection des zones de texte
        detector = ballons_modules['detector']
        text_regions = detector.detect(img_array)
        print(f"Detected {len(text_regions)} text regions")
        
        if not text_regions:
            print("No text regions detected, returning original image")
            return image
        
        print("📖 [REAL] Reconnaissance de texte...")
        
        # 2. OCR sur les zones détectées
        ocr = ballons_modules['ocr']
        extracted_texts = []
        
        for region in text_regions:
            text = ocr.recognize(img_array, region)
            if text.strip():
                extracted_texts.append((region, text))
        
        print(f"Extracted {len(extracted_texts)} text blocks")
        
        if not extracted_texts:
            print("No text extracted, returning original image")
            return image
        
        print("🌐 [REAL] Traduction...")
        
        # 3. Traduction des textes extraits
        translator = ballons_modules['translator']
        translated_texts = []
        
        for region, text in extracted_texts:
            try:
                translated = translator.translate(text, request.source_lang, request.target_lang)
                translated_texts.append((region, text, translated))
                print(f"'{text}' -> '{translated}'")
            except Exception as e:
                print(f"Translation failed for '{text}': {e}")
                translated_texts.append((region, text, text))  # Keep original if failed
        
        print("🎨 [REAL] Inpainting et rendu...")
        
        # 4. Inpainting pour effacer l'ancien texte
        inpainter = ballons_modules['inpainter']
        
        # Créer un masque pour les zones de texte
        mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
        for region, _, _ in translated_texts:
            # Dessiner le masque de la région (cette partie dépend du format de region)
            # Pour l'instant, on simule
            pass
        
        # Inpaint l'image
        inpainted_array = inpainter.inpaint(img_array, mask)
        inpainted_image = Image.fromarray(inpainted_array)
        
        # 5. Rendu du texte traduit
        draw = ImageDraw.Draw(inpainted_image)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        for region, original, translated in translated_texts:
            # Positionner le texte traduit (cette partie dépend du format de region)
            # Pour l'instant, on met le texte en haut
            y_pos = 10 + len([t for t in translated_texts if t == (region, original, translated)]) * 20
            draw.text((10, y_pos), f"{original} -> {translated}", fill="red", font=font)
        
        print("✅ Real translation complete")
        return inpainted_image
        
    except Exception as e:
        print(f"❌ Real translation failed: {e}")
        print("Falling back to simulation mode")
        return process_simulation_mode(image, request)

# Fonction de simulation (mode de fallback)
def process_simulation_mode(image, request):
    """Mode simulation - ajoute du texte pour tester"""
    print("🔍 [SIMULATION] Détection de texte...")
    time.sleep(0.5)
    
    print("📖 [SIMULATION] Reconnaissance de texte...")
    time.sleep(0.3)
    
    print("🌐 [SIMULATION] Traduction...")
    time.sleep(0.2)
    
    print("🎨 [SIMULATION] Rendu final...")
    time.sleep(0.2)
    
    # Créer une copie de l'image pour modification
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    # Ajouter du texte de simulation
    draw.text((10, 10), "TRANSLATED BY API", fill="red", font=font)
    draw.text((10, 30), f"Lang: {request.source_lang}->{request.target_lang}", fill="red", font=font)
    draw.text((10, 50), f"Size: {image.width}x{image.height}", fill="red", font=font)
    draw.text((10, 70), "Mode: SIMULATION", fill="orange", font=font)
    
    print(f"Image processed: {image.width}x{image.height}, mode: {image.mode}")
    
    return result_image

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
