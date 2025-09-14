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

# État des modules BallonsTranslator
translator_ready = False
ballons_modules = {}

# Tentative d'import des modules BallonsTranslator avec vraie structure
try:
    print("🔍 Chargement des modules BallonsTranslator...")
    
    # Imports selon la vraie structure trouvée
    from modules.translators.trans_google import GoogleTranslator
    from modules.textdetector.detector_ctd import ComicTextDetector  
    from modules.ocr.mit48px import Mit48pxOCR
    from modules.inpaint.lama import LamaInpainter
    
    print("✅ Modules BallonsTranslator importés avec succès")
    translator_ready = True
    
except ImportError as e:
    print(f"⚠️ Import des modules échoué: {e}")
    print("📋 Basculement en mode simulation")
    translator_ready = False

app = FastAPI(
    title="Manga Translator API - BallonsTranslator",
    description="API REST pour traduction d'images manga avec BallonsTranslator",
    version="1.0.0"
)

# CORS pour l'extension Chrome
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles Pydantic
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

# Utiliser lifespan au lieu de on_event (deprecated)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # Startup
    global translator_ready, ballons_modules
    
    try:
        print("🚀 Initialisation de l'API Manga Translator...")
        print(f"📁 Répertoire de travail: {os.getcwd()}")
        
        if translator_ready:
            print("🔧 Initialisation des modules BallonsTranslator...")
            
            ballons_modules = {}
            
            # Initialiser le traducteur Google
            try:
                ballons_modules['translator'] = GoogleTranslator()
                print("✅ GoogleTranslator initialisé")
            except Exception as e:
                print(f"❌ Erreur GoogleTranslator: {e}")
            
            # Initialiser le détecteur de texte
            try:
                ballons_modules['detector'] = ComicTextDetector()
                print("✅ ComicTextDetector initialisé")
            except Exception as e:
                print(f"❌ Erreur ComicTextDetector: {e}")
            
            # Initialiser l'OCR
            try:
                ballons_modules['ocr'] = Mit48pxOCR()
                print("✅ Mit48pxOCR initialisé")
            except Exception as e:
                print(f"❌ Erreur Mit48pxOCR: {e}")
            
            # Initialiser l'inpainter
            try:
                ballons_modules['inpainter'] = LamaInpainter()
                print("✅ LamaInpainter initialisé")
            except Exception as e:
                print(f"❌ Erreur LamaInpainter: {e}")
            
            print(f"🎯 {len(ballons_modules)} modules initialisés")
            
            # Si aucun module critique n'est chargé, basculer en simulation
            if not any(key in ballons_modules for key in ['translator', 'detector']):
                print("⚠️ Modules critiques manquants, mode simulation activé")
                translator_ready = False
        
        print("🎯 API prête!")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        translator_ready = False
        
    yield
    
    # Shutdown
    print("🛑 Arrêt de l'API")

app = FastAPI(
    title="Manga Translator API - BallonsTranslator",
    description="API REST pour traduction d'images manga",
    version="1.0.0",
    lifespan=lifespan
)

# Re-add middleware after app recreation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Point d'entrée principal"""
    return {
        "name": "Manga Translator API - BallonsTranslator Integration",
        "version": "1.0.0",
        "status": "running",
        "mode": "production" if translator_ready else "simulation",
        "modules_loaded": list(ballons_modules.keys()) if translator_ready else [],
        "ballons_translator_status": "integrated" if translator_ready else "fallback_mode",
        "endpoints": {
            "translate": "POST /translate",
            "translate_file": "POST /translate-file", 
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    return {
        "status": "healthy" if translator_ready else "simulation_mode",
        "ballons_translator": "loaded" if translator_ready else "not_available",
        "modules_loaded": translator_ready,
        "available_modules": list(ballons_modules.keys()) if translator_ready else [],
        "working_directory": os.getcwd(),
        "message": "BallonsTranslator integration ready" if translator_ready else "Running in simulation mode"
    }

@app.post("/translate", response_model=TranslationResponse)
async def translate_image(request: TranslationRequest):
    """
    Traduire une image manga avec BallonsTranslator
    
    Args:
        request: Données de la requête (image base64, langues, traducteur)
        
    Returns:
        TranslationResponse avec l'image traduite
    """
    start_time = time.time()
    
    try:
        # Validation et décodage de l'image
        try:
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Conversion en RGB si nécessaire
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            print(f"📸 Image reçue: {image.width}x{image.height}, mode: {image.mode}")
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image invalide: {str(e)}")
        
        # Traitement selon le mode disponible
        if translator_ready and ballons_modules:
            print("🔄 Traitement avec BallonsTranslator...")
            result_image = await process_with_ballons_translator(image, request)
        else:
            print("🔄 Traitement en mode simulation...")
            result_image = process_simulation_mode(image, request)
        
        # Conversion en base64
        buffer = io.BytesIO()
        result_image.save(buffer, format='PNG', optimize=True)
        image_bytes = buffer.getvalue()
        result_base64 = base64.b64encode(image_bytes).decode()
        
        processing_time = time.time() - start_time
        
        print(f"✅ Traitement terminé en {processing_time:.2f}s")
        
        return TranslationResponse(
            success=True,
            translated_image_base64=result_base64,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Erreur de traitement: {e}")
        
        return TranslationResponse(
            success=False,
            error=str(e),
            processing_time=processing_time
        )

@app.post("/translate-file")
async def translate_file(file: UploadFile = File(...)):
    """
    Upload direct d'un fichier image
    """
    try:
        # Validation du type de fichier
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Le fichier doit être une image")
        
        # Lire le fichier
        image_data = await file.read()
        image_base64 = base64.b64encode(image_data).decode()
        
        # Traiter via l'endpoint principal
        request = TranslationRequest(image_base64=image_base64)
        return await translate_image(request)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de traitement: {str(e)}")

# Traitement réel avec BallonsTranslator
async def process_with_ballons_translator(image, request):
    """Traitement avec les vrais modules BallonsTranslator"""
    try:
        print("🔍 Détection des zones de texte...")
        
        # Conversion PIL -> numpy pour les modules
        img_array = np.array(image)
        
        # 1. Détection des zones de texte
        if 'detector' not in ballons_modules:
            print("⚠️ Détecteur non disponible, retour au mode simulation")
            return process_simulation_mode(image, request)
        
        detector = ballons_modules['detector']
        
        # Utiliser l'API du détecteur BallonsTranslator
        try:
            # La méthode exacte dépend de l'implémentation de detector_ctd.py
            text_regions = detector.detect(img_array)
            print(f"📍 {len(text_regions)} zones de texte détectées")
        except Exception as e:
            print(f"❌ Erreur de détection: {e}")
            return add_debug_info(image, f"Erreur détection: {str(e)}")
        
        if not text_regions:
            return add_debug_info(image, "Aucune zone de texte détectée")
        
        # 2. OCR sur les zones détectées
        print("📖 Reconnaissance de texte...")
        
        extracted_texts = []
        
        if 'ocr' in ballons_modules:
            ocr = ballons_modules['ocr']
            
            try:
                for i, region in enumerate(text_regions):
                    # Adapter selon l'API de l'OCR BallonsTranslator
                    text = ocr.recognize(img_array, region)
                    if text and text.strip():
                        extracted_texts.append((region, text.strip()))
                        print(f"📝 Texte {i+1}: '{text.strip()}'")
            except Exception as e:
                print(f"❌ Erreur OCR: {e}")
                return add_debug_info(image, f"Erreur OCR: {str(e)}")
        
        if not extracted_texts:
            return add_debug_info(image, f"Aucun texte extrait de {len(text_regions)} zones")
        
        # 3. Traduction des textes
        print("🌐 Traduction des textes...")
        
        translated_texts = []
        
        if 'translator' in ballons_modules:
            translator = ballons_modules['translator']
            
            try:
                for region, text in extracted_texts:
                    # Utiliser l'API du traducteur BallonsTranslator
                    translated = translator.translate(text, target_language=request.target_lang)
                    translated_texts.append((region, text, translated))
                    print(f"🔄 '{text}' -> '{translated}'")
            except Exception as e:
                print(f"❌ Erreur traduction: {e}")
                # Garder le texte original en cas d'erreur
                for region, text in extracted_texts:
                    translated_texts.append((region, text, f"[ERREUR] {text}"))
        
        # 4. Rendu final
        print("🎨 Composition de l'image finale...")
        
        # Essayer l'inpainting si disponible
        if 'inpainter' in ballons_modules and len(translated_texts) > 0:
            try:
                return render_with_inpainting(image, img_array, translated_texts)
            except Exception as e:
                print(f"⚠️ Inpainting échoué: {e}, rendu simple")
                return render_ballons_data(image, translated_texts)
        else:
            return render_ballons_data(image, translated_texts)
        
    except Exception as e:
        print(f"❌ Erreur dans le traitement BallonsTranslator: {e}")
        import traceback
        traceback.print_exc()
        return add_debug_info(image, f"Erreur critique: {str(e)}")

def render_with_inpainting(original_image, img_array, translated_texts):
    """Rendu avec inpainting BallonsTranslator"""
    try:
        inpainter = ballons_modules['inpainter']
        
        # Créer un masque pour les zones de texte
        mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
        
        # Cette partie dépend du format exact des régions retournées par le détecteur
        for region, _, _ in translated_texts:
            # Adapter selon la structure des régions de BallonsTranslator
            try:
                if hasattr(region, 'bbox') or hasattr(region, 'xyxy'):
                    # Extraire les coordonnées selon le format
                    if hasattr(region, 'bbox'):
                        x, y, w, h = region.bbox
                    else:
                        x1, y1, x2, y2 = region.xyxy
                        x, y, w, h = x1, y1, x2-x1, y2-y1
                    
                    # Remplir le masque
                    mask[max(0, int(y)):min(mask.shape[0], int(y+h)), 
                         max(0, int(x)):min(mask.shape[1], int(x+w))] = 255
                         
            except Exception as e:
                print(f"⚠️ Erreur création masque pour région: {e}")
        
        # Appliquer l'inpainting
        inpainted_array = inpainter.inpaint(img_array, mask)
        result_image = Image.fromarray(inpainted_array.astype(np.uint8))
        
        # Ajouter le texte traduit
        draw = ImageDraw.Draw(result_image)
        font = get_font()
        
        for region, original, translated in translated_texts:
            try:
                # Positionner le texte traduit
                if hasattr(region, 'bbox'):
                    x, y, w, h = region.bbox
                elif hasattr(region, 'xyxy'):
                    x, y = region.xyxy[:2]
                else:
                    x, y = 10, 10  # Position par défaut
                
                # Fond pour le texte
                text_bbox = draw.textbbox((x, y), translated, font=font)
                draw.rectangle(text_bbox, fill=(255, 255, 255, 200))
                
                # Texte traduit
                draw.text((x, y), translated, fill="black", font=font)
                
            except Exception as e:
                print(f"⚠️ Erreur rendu texte: {e}")
        
        return result_image
        
    except Exception as e:
        print(f"❌ Erreur inpainting: {e}")
        return render_ballons_data(original_image, translated_texts)

def render_ballons_data(image, translated_texts):
    """Rendu simple avec données BallonsTranslator réelles"""
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    font = get_font()
    
    # Overlay vert pour indiquer données réelles
    y_offset = 10
    
    # Titre
    draw.text((10, y_offset), "🎯 BALLONSTRANSLATOR DATA", fill="green", font=font)
    y_offset += 25
    
    # Afficher les traductions réelles
    for i, (region, original, translated) in enumerate(translated_texts[:5]):  # Limiter l'affichage
        text = f"{i+1}. '{original}' -> '{translated}'"
        
        # Fond semi-transparent
        text_bbox = draw.textbbox((10, y_offset), text, font=font)
        draw.rectangle([(text_bbox[0]-2, text_bbox[1]-2), (text_bbox[2]+2, text_bbox[3]+2)], 
                      fill=(0, 0, 0, 128))
        
        # Texte
        draw.text((10, y_offset), text, fill="white", font=font)
        y_offset += 20
    
    # Info technique
    if len(translated_texts) > 5:
        draw.text((10, y_offset + 5), f"... et {len(translated_texts)-5} autres", fill="gray", font=font)
    
    return result_image

def add_debug_info(image, message):
    """Ajouter info de debug à l'image"""
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    font = get_font()
    
    draw.text((10, 10), f"DEBUG: {message}", fill="red", font=font)
    draw.text((10, 30), f"Mode: BallonsTranslator (debug)", fill="blue", font=font)
    
    return result_image

def process_simulation_mode(image, request):
    """Mode simulation amélioré"""
    print("🎭 Mode simulation...")
    
    # Simulation réaliste des étapes
    time.sleep(0.2)
    
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    font = get_font()
    
    # Titre simulation
    draw.text((10, 10), "🎭 SIMULATION MODE", fill="orange", font=font)
    
    # Simuler des traductions
    simulated_translations = [
        ("こんにちは", "Hello"),
        ("ありがとう", "Thank you"), 
        ("元気ですか", "How are you?"),
        ("さようなら", "Goodbye")
    ]
    
    y_offset = 35
    for i, (original, translated) in enumerate(simulated_translations[:3]):
        text = f"{i+1}. '{original}' -> '{translated}'"
        draw.text((10, y_offset), text, fill="red", font=font)
        y_offset += 18
    
    # Info technique
    draw.text((10, y_offset + 10), f"Taille: {image.width}x{image.height}", fill="gray", font=font)
    draw.text((10, y_offset + 25), f"{request.source_lang} -> {request.target_lang}", fill="blue", font=font)
    
    return result_image

def get_font():
    """Obtenir une police pour le rendu de texte"""
    try:
        # Essayer une police système
        return ImageFont.truetype("arial.ttf", 14)
    except:
        try:
            return ImageFont.load_default()
        except:
            return None

if __name__ == "__main__":
    print("🚀 Démarrage Manga Translator API avec BallonsTranslator...")
    print("📚 Documentation: http://localhost:8000/docs")
    print("💚 Health check: http://localhost:8000/health")
    print("🎯 Interface: http://localhost:8000/")
    
    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
