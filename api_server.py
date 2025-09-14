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
import cv2

# Ajouter le chemin vers BallonsTranslator
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# État des modules BallonsTranslator
translator_ready = False
ballons_modules = {}

# Tentative d'import des modules BallonsTranslator
try:
    print("🔍 Recherche des modules BallonsTranslator...")
    
    # Vérification de la structure du projet
    modules_path = os.path.join(current_dir, 'modules')
    if not os.path.exists(modules_path):
        print(f"❌ Dossier modules non trouvé: {modules_path}")
        raise ImportError("Modules directory not found")
    
    # Import conditionnel selon la vraie structure
    try:
        # Essayer les imports principaux
        from modules.translators.google import GoogleTranslator
        from modules.textdetector.ctd import ComicTextDetector
        from modules.ocr.mit48px_ctd import Mit48pxCTDOCR
        from modules.inpaint.lama_large_512px import LamaLargeInpainter
        
        print("✅ Modules BallonsTranslator importés avec succès")
        translator_ready = True
        
    except ImportError as e1:
        print(f"⚠️ Import principal échoué: {e1}")
        
        # Essayer une structure alternative
        try:
            from translators.google import GoogleTranslator
            from textdetector.ctd import ComicTextDetector
            from ocr.mit48px_ctd import Mit48pxCTDOCR
            from inpaint.lama_large_512px import LamaLargeInpainter
            
            print("✅ Modules BallonsTranslator importés (structure alternative)")
            translator_ready = True
            
        except ImportError as e2:
            print(f"⚠️ Structure alternative échouée: {e2}")
            raise ImportError("Could not import BallonsTranslator modules")
    
except ImportError as e:
    print(f"❌ Impossible d'importer BallonsTranslator: {e}")
    print("📋 Mode simulation activé")
    translator_ready = False

app = FastAPI(
    title="Manga Translator API",
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

@app.on_event("startup")
async def startup_event():
    """Initialisation des modules au démarrage"""
    global translator_ready, ballons_modules
    
    try:
        print("🚀 Initialisation de l'API Manga Translator...")
        print(f"📁 Répertoire de travail: {os.getcwd()}")
        
        if translator_ready:
            print("🔧 Chargement des modules BallonsTranslator...")
            
            # Initialiser les modules avec gestion d'erreur
            ballons_modules = {}
            
            try:
                ballons_modules['translator'] = GoogleTranslator()
                print("✅ GoogleTranslator initialisé")
            except Exception as e:
                print(f"❌ Erreur GoogleTranslator: {e}")
            
            try:
                ballons_modules['detector'] = ComicTextDetector()
                print("✅ ComicTextDetector initialisé")
            except Exception as e:
                print(f"❌ Erreur ComicTextDetector: {e}")
            
            try:
                ballons_modules['ocr'] = Mit48pxCTDOCR()
                print("✅ Mit48pxCTDOCR initialisé")
            except Exception as e:
                print(f"❌ Erreur OCR: {e}")
            
            try:
                ballons_modules['inpainter'] = LamaLargeInpainter()
                print("✅ LamaLargeInpainter initialisé")
            except Exception as e:
                print(f"❌ Erreur Inpainter: {e}")
            
            # Vérifier qu'au moins certains modules sont chargés
            if len(ballons_modules) >= 2:
                print(f"✅ {len(ballons_modules)} modules chargés avec succès")
            else:
                print("⚠️ Pas assez de modules chargés, passage en mode simulation")
                translator_ready = False
        
        print("🎯 API prête!")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        print("📋 Basculement en mode simulation")
        translator_ready = False

@app.get("/")
async def root():
    """Point d'entrée principal"""
    return {
        "name": "Manga Translator API",
        "version": "1.0.0",
        "status": "running",
        "mode": "production" if translator_ready else "simulation",
        "modules_loaded": list(ballons_modules.keys()) if translator_ready else [],
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
        "modules_loaded": translator_ready,
        "available_modules": list(ballons_modules.keys()) if translator_ready else [],
        "working_directory": os.getcwd(),
        "python_path": sys.path[:3]  # Premiers éléments du path
    }

@app.post("/translate", response_model=TranslationResponse)
async def translate_image(request: TranslationRequest):
    """
    Traduire une image manga
    
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
            print("⚠️ Détecteur non disponible, mode simulation partiel")
            return process_simulation_mode(image, request)
        
        detector = ballons_modules['detector']
        
        # Adapter selon l'API réelle du détecteur
        try:
            text_regions = detector.detect_text(img_array)
        except AttributeError:
            try:
                text_regions = detector(img_array)
            except:
                text_regions = []
        
        print(f"📍 {len(text_regions)} zones de texte détectées")
        
        if not text_regions:
            print("❌ Aucune zone de texte trouvée")
            return add_debug_info(image, "Aucun texte détecté")
        
        # 2. OCR sur les zones détectées
        print("📖 Reconnaissance de texte...")
        
        extracted_texts = []
        
        if 'ocr' in ballons_modules:
            ocr = ballons_modules['ocr']
            
            for region in text_regions:
                try:
                    text = ocr.recognize(img_array, region)
                    if text and text.strip():
                        extracted_texts.append((region, text.strip()))
                        print(f"📝 Texte extrait: '{text.strip()}'")
                except Exception as e:
                    print(f"❌ Erreur OCR sur région: {e}")
        
        if not extracted_texts:
            return add_debug_info(image, f"OCR: 0 textes extraits de {len(text_regions)} zones")
        
        # 3. Traduction des textes
        print("🌐 Traduction des textes...")
        
        translated_texts = []
        
        if 'translator' in ballons_modules:
            translator = ballons_modules['translator']
            
            for region, text in extracted_texts:
                try:
                    translated = translator.translate(text, request.source_lang, request.target_lang)
                    translated_texts.append((region, text, translated))
                    print(f"🔄 '{text}' -> '{translated}'")
                except Exception as e:
                    print(f"❌ Erreur traduction pour '{text}': {e}")
                    translated_texts.append((region, text, f"[ERREUR: {text}]"))
        
        # 4. Rendu final
        print("🎨 Composition de l'image finale...")
        
        # Si inpainter disponible, l'utiliser, sinon simulation avancée
        if 'inpainter' in ballons_modules and len(translated_texts) > 0:
            return render_with_inpainting(image, img_array, translated_texts)
        else:
            return render_simulation_with_real_data(image, translated_texts)
        
    except Exception as e:
        print(f"❌ Erreur dans le traitement BallonsTranslator: {e}")
        return add_debug_info(image, f"Erreur: {str(e)}")

def render_with_inpainting(original_image, img_array, translated_texts):
    """Rendu avec inpainting réel"""
    try:
        inpainter = ballons_modules['inpainter']
        
        # Créer un masque pour les zones de texte
        mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
        
        # Remplir le masque (dépend du format des régions)
        for region, _, _ in translated_texts:
            # Cette partie dépend du format exact des régions retournées
            # par le détecteur - à adapter selon l'API réelle
            if hasattr(region, 'bbox'):
                x, y, w, h = region.bbox
                mask[y:y+h, x:x+w] = 255
        
        # Inpainting
        inpainted_array = inpainter.inpaint(img_array, mask)
        result_image = Image.fromarray(inpainted_array)
        
        # Ajouter le texte traduit
        draw = ImageDraw.Draw(result_image)
        font = get_font()
        
        for region, original, translated in translated_texts:
            if hasattr(region, 'bbox'):
                x, y, w, h = region.bbox
                draw.text((x, y), translated, fill="black", font=font)
        
        return result_image
        
    except Exception as e:
        print(f"❌ Erreur inpainting: {e}")
        return render_simulation_with_real_data(original_image, translated_texts)

def render_simulation_with_real_data(image, translated_texts):
    """Rendu simulation mais avec vraies données détectées"""
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    font = get_font()
    
    # Ajouter les traductions avec overlay coloré
    y_offset = 10
    
    for region, original, translated in translated_texts:
        # Fond semi-transparent pour le texte
        text_bbox = draw.textbbox((10, y_offset), f"'{original}' -> '{translated}'", font=font)
        draw.rectangle(text_bbox, fill=(0, 0, 0, 128))
        
        # Texte traduit
        draw.text((10, y_offset), f"'{original}' -> '{translated}'", fill="white", font=font)
        y_offset += 25
    
    # Info de debug
    draw.text((10, y_offset + 10), f"REAL DATA: {len(translated_texts)} traductions", fill="green", font=font)
    
    return result_image

def add_debug_info(image, message):
    """Ajouter info de debug à l'image"""
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    font = get_font()
    
    draw.text((10, 10), f"DEBUG: {message}", fill="red", font=font)
    draw.text((10, 30), f"Mode: BallonsTranslator", fill="blue", font=font)
    
    return result_image

def process_simulation_mode(image, request):
    """Mode simulation amélioré"""
    print("🎭 Mode simulation - traitement factice...")
    
    # Simulation réaliste des étapes
    time.sleep(0.3)  # Détection
    time.sleep(0.2)  # OCR
    time.sleep(0.1)  # Traduction
    
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    font = get_font()
    
    # Simuler des traductions
    simulated_translations = [
        ("こんにちは", "Hello"),
        ("ありがとう", "Thank you"),
        ("元気ですか", "How are you?")
    ]
    
    y_offset = 10
    for original, translated in simulated_translations:
        draw.text((10, y_offset), f"'{original}' -> '{translated}'", fill="red", font=font)
        y_offset += 20
    
    # Informations du traitement
    draw.text((10, y_offset + 10), f"SIMULATION MODE", fill="orange", font=font)
    draw.text((10, y_offset + 30), f"{request.source_lang} -> {request.target_lang}", fill="blue", font=font)
    draw.text((10, y_offset + 50), f"Taille: {image.width}x{image.height}", fill="gray", font=font)
    
    return result_image

def get_font():
    """Obtenir une police pour le rendu de texte"""
    try:
        # Essayer une police système
        return ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            return ImageFont.load_default()
        except:
            return None

if __name__ == "__main__":
    print("🚀 Démarrage du serveur Manga Translator API...")
    print("📚 Documentation: http://localhost:8000/docs")
    print("💚 Health check: http://localhost:8000/health")
    print("🔧 Panel admin: http://localhost:8000/")
    
    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
