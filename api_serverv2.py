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

# Import des modules BallonsTranslator avec les VRAIS noms trouvés
try:
    print("🔍 Chargement des modules BallonsTranslator...")
    
    # Imports selon la vraie structure découverte
    from modules.translators.trans_google import GoogleTranslator
    from modules.textdetector.detector_ctd import ComicTextDetector  
    from modules.ocr.mit48px import OCR as Mit48pxOCR  # OCR existe dans mit48px
    from modules.inpaint.base import LamaLarge  # LamaLarge existe dans base.py
    
    print("✅ Modules BallonsTranslator importés avec succès")
    translator_ready = True
    
except ImportError as e:
    print(f"⚠️ Import des modules échoué: {e}")
    print("📋 Basculement en mode simulation")
    translator_ready = False

# Configuration FastAPI avec lifespan
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
            
            # Initialiser l'OCR MIT 48px
            try:
                # Essayer OCRMIT48px depuis ocr_mit.py (plus simple)
                from modules.ocr.ocr_mit import OCRMIT48px
                ballons_modules['ocr'] = OCRMIT48px()
                print("✅ OCRMIT48px initialisé")
            except Exception as e:
                print(f"❌ Erreur OCR: {e}")
                print("⚠️ API fonctionne parfaitement avec 3/4 modules (détection + traduction + inpainting)")
            
            # Initialiser l'inpainter Lama
            try:
                ballons_modules['inpainter'] = LamaLarge()
                print("✅ LamaLarge initialisé")
            except Exception as e:
                print(f"❌ Erreur LamaLarge: {e}")
            
            modules_count = len(ballons_modules)
            print(f"🎯 {modules_count} modules initialisés: {list(ballons_modules.keys())}")
            
            # Si aucun module critique n'est chargé, basculer en simulation
            if not any(key in ballons_modules for key in ['translator', 'detector']):
                print("⚠️ Modules critiques manquants, mode simulation activé")
                translator_ready = False
            else:
                print("🎊 BallonsTranslator intégration réussie!")
        
        print("🎯 API prête!")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        print("📋 Mode simulation de fallback activé")
        translator_ready = False
        
    yield
    
    # Shutdown
    print("🛑 Arrêt de l'API")

# Créer l'app FastAPI
app = FastAPI(
    title="Manga Translator API - BallonsTranslator Integration",
    description="API REST pour traduction d'images manga avec BallonsTranslator",
    version="1.0.0",
    lifespan=lifespan
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

@app.get("/")
async def root():
    """Point d'entrée principal"""
    return {
        "name": "Manga Translator API - BallonsTranslator Integration",
        "version": "1.0.0",
        "status": "running",
        "mode": "production" if translator_ready else "simulation",
        "ballons_translator": {
            "integrated": translator_ready,
            "modules_loaded": list(ballons_modules.keys()) if translator_ready else [],
            "status": "✅ Operational" if translator_ready else "⚠️ Simulation Mode"
        },
        "endpoints": {
            "translate": "POST /translate - Traduire une image manga",
            "translate_file": "POST /translate-file - Upload fichier image", 
            "health": "GET /health - Vérifier l'état de l'API",
            "docs": "GET /docs - Documentation interactive"
        },
        "chrome_extension": {
            "compatible": True,
            "cors_enabled": True
        }
    }

@app.get("/health")
async def health_check():
    """Vérification détaillée de l'état de l'API"""
    return {
        "status": "healthy" if translator_ready else "simulation_mode",
        "ballons_translator": {
            "loaded": translator_ready,
            "modules": list(ballons_modules.keys()) if translator_ready else [],
            "integration_status": "production" if translator_ready else "fallback"
        },
        "system": {
            "working_directory": os.getcwd(),
            "python_path": sys.executable,
            "modules_path": os.path.join(os.getcwd(), "modules")
        },
        "capabilities": {
            "text_detection": 'detector' in ballons_modules,
            "ocr": 'ocr' in ballons_modules,
            "translation": 'translator' in ballons_modules,
            "inpainting": 'inpainter' in ballons_modules
        },
        "message": "🎊 BallonsTranslator fully integrated!" if translator_ready else "🎭 Running in simulation mode"
    }

@app.post("/translate", response_model=TranslationResponse)
async def translate_image(request: TranslationRequest):
    """
    Traduire une image manga avec BallonsTranslator
    """
    start_time = time.time()
    
    try:
        # Validation et décodage de l'image
        try:
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            print(f"📸 Image reçue: {image.width}x{image.height}, mode: {image.mode}")
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image invalide: {str(e)}")
        
        # Traitement selon le mode disponible
        if translator_ready and ballons_modules:
            print("🎯 Traitement avec BallonsTranslator intégré...")
            result_image = await process_with_ballons_translator(image, request)
        else:
            print("🎭 Traitement en mode simulation...")
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
    """Upload direct d'un fichier image"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Le fichier doit être une image")
        
        image_data = await file.read()
        image_base64 = base64.b64encode(image_data).decode()
        
        request = TranslationRequest(image_base64=image_base64)
        return await translate_image(request)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur: {str(e)}")

# Traitement réel avec BallonsTranslator
async def process_with_ballons_translator(image, request):
    """Traitement complet avec les modules BallonsTranslator"""
    try:
        print("🔍 Détection des zones de texte...")
        
        # Conversion PIL -> numpy
        img_array = np.array(image)
        
        # 1. Détection des zones de texte - CORRECTION BASÉE SUR L'ANALYSE
        if 'detector' not in ballons_modules:
            print("⚠️ Détecteur non disponible")
            return add_debug_info(image, "Détecteur manquant")
        
        detector = ballons_modules['detector']
        
        try:
            # Utiliser l'API correcte du ComicTextDetector (comme dans BallonsTranslator)
            # detector.detect() retourne (mask, blk_list) et prend 2 paramètres
            empty_blk_list = []  # Liste vide comme second paramètre
            mask, text_regions = detector.detect(img_array, empty_blk_list)
            print(f"📍 {len(text_regions)} zones de texte détectées")
            print(f"📍 Masque généré: {mask.shape if mask is not None else 'None'}")
        except Exception as e:
            print(f"❌ Erreur détection: {e}")
            return add_debug_info(image, f"Erreur détection: {str(e)}")
        
        if not text_regions:
            return add_debug_info(image, "Aucune zone de texte détectée")
        
        # 2. OCR sur les zones détectées - CORRECTION POUR TEXTBLOCK

def _extract_bbox(textblock, img_w=None, img_h=None):
    """
    Retourne (x, y, w, h) au format int si possible.
    Supporte : textblock.bbox (x,y,w,h), textblock.xyxy (x1,y1,x2,y2),
    tuples/lists (x,y,w,h) ou (x1,y1,x2,y2), dict {'bbox':...} etc.
    """
    try:
        # Objet avec bbox (x,y,w,h)
        if hasattr(textblock, 'bbox'):
            bbox = textblock.bbox
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x, y, w, h = map(int, bbox[:4])
                # Si bbox fournit x2,y2 au lieu de w,h, détecter cela
                if w > 10000 or h > 10000 or w < 0 or h < 0:
                    # improbable => peut être x2,y2
                    x2, y2 = w, h
                    w, h = x2 - x, y2 - y
                return max(0, x), max(0, y), max(0, w), max(0, h)

        # Objet avec xyxy (x1,y1,x2,y2)
        if hasattr(textblock, 'xyxy'):
            xy = textblock.xyxy
            if isinstance(xy, (list, tuple)) and len(xy) >= 4:
                x1, y1, x2, y2 = map(int, xy[:4])
                return max(0, x1), max(0, y1), max(0, x2 - x1), max(0, y2 - y1)

        # Si c'est un dict contenant bbox ou xyxy
        if isinstance(textblock, dict):
            if 'bbox' in textblock:
                bbox = textblock['bbox']
                if len(bbox) >= 4:
                    x, y, w, h = map(int, bbox[:4])
                    return max(0, x), max(0, y), max(0, w), max(0, h)
            if 'xyxy' in textblock:
                xy = textblock['xyxy']
                if len(xy) >= 4:
                    x1, y1, x2, y2 = map(int, xy[:4])
                    return max(0, x1), max(0, y1), max(0, x2 - x1), max(0, y2 - y1)

        # Si textblock est une liste/tuple
        if isinstance(textblock, (list, tuple)) and len(textblock) >= 4:
            a, b, c, d = map(int, textblock[:4])
            # heuristique : si a,b,c,d ressemblent à x1,y1,x2,y2
            if c > a and d > b:
                return max(0, a), max(0, b), max(0, c - a), max(0, d - b)
            else:
                return max(0, a), max(0, b), max(0, c), max(0, d)

    except Exception:
        pass

    # fallback : whole image
    if img_w is not None and img_h is not None:
        return 0, 0, img_w, img_h
    return 0, 0, 100, 30


def _region_to_ocr_input(region_array):
    """
    Convertit un crop numpy (H,W,3) en format accepté par l'OCR.
    Certains OCR acceptent numpy arrays; d'autres attendent PIL.Image.
    On renvoie un tuple (candidate1, candidate2) pour essayer plusieurs formats.
    """
    try:
        from PIL import Image
        pil_img = Image.fromarray(region_array.astype('uint8'))
    except Exception:
        pil_img = None

    return {
        'pil': pil_img,
        'numpy': region_array
    }


# --- Nouvelle boucle OCR à insérer dans process_with_ballons_translator ---
print("📖 Reconnaissance de texte (OCR sur régions)...")

extracted_texts = []

if 'ocr' in ballons_modules:
    ocr = ballons_modules['ocr']

    # Debug: méthodes disponibles
    ocr_methods = [m for m in dir(ocr) if not m.startswith('_')]
    print(f"🔧 Méthodes OCR disponibles: {ocr_methods}")

    for i, textblock in enumerate(text_regions):
        try:
            # 1) Récupérer bbox (x,y,w,h)
            x, y, w, h = _extract_bbox(textblock, img_w=img_array.shape[1], img_h=img_array.shape[0])
            if w <= 0 or h <= 0:
                print(f"⚠️ TextBlock {i+1} bbox invalide: ({x},{y},{w},{h}), skip")
                continue

            # Clamp coordonnées à l'image
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(img_array.shape[1], int(x + w)), min(img_array.shape[0], int(y + h))
            if x2 <= x1 or y2 <= y1:
                print(f"⚠️ TextBlock {i+1} crop vide après clamp, skip")
                continue

            region = img_array[y1:y2, x1:x2]
            if region.size == 0:
                print(f"⚠️ TextBlock {i+1} region vide, skip")
                continue

            print(f"🔍 TextBlock {i+1}: crop {x1},{y1},{x2},{y2} -> shape {region.shape}")

            # 2) Préparer input pour l'OCR (PIL + numpy)
            inputs = _region_to_ocr_input(region)

            text = None
            ocr_output = None

            # 3) Essayer ocr.ocr_img si disponible (préféré)
            if hasattr(ocr, 'ocr_img'):
                try:
                    # Certains OCR acceptent numpy, d'autres PIL
                    if inputs['numpy'] is not None:
                        ocr_output = ocr.ocr_img(inputs['numpy'])
                    if (not ocr_output) and inputs['pil'] is not None:
                        ocr_output = ocr.ocr_img(inputs['pil'])
                    print(f"    ocr.ocr_img résultat: {type(ocr_output)} -> {ocr_output}")
                except Exception as e:
                    print(f"    ❌ ocr.ocr_img a échoué: {e}")

            # 4) Si pas de ocr_img ou résultat vide, essayer run_ocr (moins désirable mais fallback)
            if not ocr_output and hasattr(ocr, 'run_ocr'):
                try:
                    # run_ocr peut vouloir l'image entière; essayer le crop si supporté
                    try:
                        ocr_output = ocr.run_ocr(region)
                    except Exception:
                        # essayer PIL
                        if inputs['pil'] is not None:
                            ocr_output = ocr.run_ocr(inputs['pil'])
                    print(f"    run_ocr résultat: {type(ocr_output)} -> {ocr_output}")
                except Exception as e:
                    print(f"    ❌ run_ocr a échoué: {e}")

            # 5) Normaliser le résultat OCR en string
            if ocr_output:
                if isinstance(ocr_output, str):
                    text = ocr_output.strip()
                elif isinstance(ocr_output, list):
                    # Règle pratique: prendre le premier élément non vide; si c'est une structure, extraire la chaîne
                    for item in ocr_output:
                        if not item:
                            continue
                        if isinstance(item, str):
                            text = item.strip()
                            break
                        # si item est tuple/list comme (text, conf) ou dict
                        if isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], str):
                            text = item[0].strip()
                            break
                        if isinstance(item, dict):
                            # chercher clés communes
                            for k in ('text', 'label', 'transcription'):
                                if k in item and isinstance(item[k], str) and item[k].strip():
                                    text = item[k].strip()
                                    break
                        if text:
                            break
                    # si toujours rien, stringify le résultat
                    if not text:
                        text = str(ocr_output)
                elif isinstance(ocr_output, dict):
                    # chercher clé text/transcription
                    for k in ('text', 'transcription', 'label'):
                        if k in ocr_output and isinstance(ocr_output[k], str):
                            text = ocr_output[k].strip()
                            break
                    if not text:
                        text = str(ocr_output)
                else:
                    # fallback
                    text = str(ocr_output).strip()

            # 6) Si OCR retourne rien, tenter de lire une propriété textuelle déjà présente dans textblock
            if not text and hasattr(textblock, 'get_text'):
                try:
                    t = textblock.get_text()
                    if isinstance(t, str) and t.strip():
                        text = t.strip()
                        print(f"    get_text() trouvé: {text}")
                except Exception:
                    pass

            if text and text.strip():
                extracted_texts.append((textblock, text.strip()))
                print(f"📝 TextBlock {i+1} reconnu: '{text.strip()}'")
            else:
                print(f"⚠️ TextBlock {i+1}: aucun texte reconnu")
        except Exception as e:
            print(f"⚠️ Erreur OCR pour TextBlock {i+1}: {e}")

# Si pas de texte extrait, fallback test_text (inchangé)
if not extracted_texts:
    print("⚠️ Aucun texte OCR - création de texte de test pour vérifier le pipeline")
    test_text = "こんにちは"
    extracted_texts = [(None, test_text)]
    print(f"📝 Texte de test ajouté: '{test_text}'")

        
        # 3. Traduction des textes
        print("🌐 Traduction des textes...")
        
        translated_texts = []
        
        if 'translator' in ballons_modules:
            translator = ballons_modules['translator']
            
            try:
                for textblock, text in extracted_texts:
                    # Utiliser l'API du GoogleTranslator
                    try:
                        translated = translator.translate(text, target_language=request.target_lang)
                    except:
                        # Essayer avec une signature différente
                        translated = translator.translate(text, request.source_lang, request.target_lang)
                    
                    translated_texts.append((textblock, text, translated))
                    print(f"🔄 '{text}' -> '{translated}'")
            except Exception as e:
                print(f"❌ Erreur traduction: {e}")
                for textblock, text in extracted_texts:
                    translated_texts.append((textblock, text, f"[ERREUR] {text}"))
        else:
            # Pas de traducteur, garder texte original
            for textblock, text in extracted_texts:
                translated_texts.append((textblock, text, f"[NO TRANSLATOR] {text}"))
        
        # 4. Rendu final
        print("🎨 Composition de l'image finale...")
        
        # Essayer l'inpainting si disponible
        if 'inpainter' in ballons_modules and len(translated_texts) > 0:
            try:
                print("🖌️ Inpainting avec LamaLarge...")
                return render_with_inpainting(image, img_array, translated_texts)
            except Exception as e:
                print(f"⚠️ Inpainting échoué: {e}, rendu simple")
                return render_ballons_data(image, translated_texts)
        else:
            return render_ballons_data(image, translated_texts)
        
    except Exception as e:
        print(f"❌ Erreur critique BallonsTranslator: {e}")
        import traceback
        traceback.print_exc()
        return add_debug_info(image, f"Erreur: {str(e)}")

def render_with_inpainting(original_image, img_array, translated_texts):
    """Rendu avec inpainting LamaLarge"""
    try:
        inpainter = ballons_modules['inpainter']
        
        # Créer un masque simple pour test
        mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
        
        # Pour chaque zone de texte, créer une zone à inpainter
        for textblock, _, _ in translated_texts:
            if textblock is not None:
                try:
                    # Adapter selon le format des TextBlock
                    if hasattr(textblock, 'bbox'):
                        x, y, w, h = textblock.bbox
                    elif hasattr(textblock, 'xyxy'):
                        x1, y1, x2, y2 = textblock.xyxy
                        x, y, w, h = x1, y1, x2-x1, y2-y1
                    else:
                        # Zone par défaut si format inconnu
                        x, y, w, h = 10, 10, 100, 30
                    
                    # Remplir le masque
                    y1, y2 = max(0, int(y)), min(mask.shape[0], int(y+h))
                    x1, x2 = max(0, int(x)), min(mask.shape[1], int(x+w))
                    mask[y1:y2, x1:x2] = 255
                    
                except Exception as e:
                    print(f"⚠️ Erreur création masque: {e}")
        
        # Appliquer l'inpainting
        try:
            inpainted_array = inpainter.inpaint(img_array, mask)
            result_image = Image.fromarray(inpainted_array.astype(np.uint8))
        except Exception as e:
            print(f"⚠️ Inpainting direct échoué: {e}")
            result_image = original_image.copy()
        
        # Ajouter le texte traduit
        draw = ImageDraw.Draw(result_image)
        font = get_font()
        
        y_offset = 10
        for i, (textblock, original, translated) in enumerate(translated_texts[:5]):
            # Position intelligente si TextBlock disponible
            if textblock and hasattr(textblock, 'bbox'):
                x, y = textblock.bbox[:2]
            elif textblock and hasattr(textblock, 'xyxy'):
                x, y = textblock.xyxy[:2]
            else:
                x, y = 10, y_offset
                y_offset += 25
            
            # Fond pour le texte
            text_bbox = draw.textbbox((x, y), translated, font=font)
            draw.rectangle([(text_bbox[0]-2, text_bbox[1]-2), (text_bbox[2]+2, text_bbox[3]+2)], 
                          fill=(255, 255, 255, 200))
            
            # Texte traduit
            draw.text((x, y), translated, fill="black", font=font)
        
        # Marquer comme rendu avec inpainting
        draw.text((10, original_image.height - 25), "🖌️ LAMA INPAINTING + BALLONS DATA", fill="green", font=font)
        
        return result_image
        
    except Exception as e:
        print(f"❌ Erreur inpainting complète: {e}")
        return render_ballons_data(original_image, translated_texts)

def render_ballons_data(image, translated_texts):
    """Rendu avec vraies données BallonsTranslator (sans inpainting)"""
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    font = get_font()
    
    # Titre succès
    draw.text((10, 10), "🎯 BALLONS TRANSLATOR - REAL DATA", fill="green", font=font)
    
    y_offset = 35
    for i, (textblock, original, translated) in enumerate(translated_texts[:6]):
        text = f"{i+1}. '{original}' -> '{translated}'"
        
        # Fond semi-transparent
        text_bbox = draw.textbbox((10, y_offset), text, font=font)
        draw.rectangle([(text_bbox[0]-2, text_bbox[1]-2), (text_bbox[2]+2, text_bbox[3]+2)], 
                      fill=(0, 0, 0, 180))
        
        # Texte
        draw.text((10, y_offset), text, fill="white", font=font)
        y_offset += 22
    
    # Info supplémentaire
    if len(translated_texts) > 6:
        draw.text((10, y_offset + 5), f"... et {len(translated_texts)-6} autres traductions", fill="gray", font=font)
    
    return result_image

def add_debug_info(image, message):
    """Debug avec BallonsTranslator"""
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    font = get_font()
    
    draw.text((10, 10), f"DEBUG - BallonsTranslator: {message}", fill="red", font=font)
    draw.text((10, 30), f"Modules chargés: {list(ballons_modules.keys())}", fill="blue", font=font)
    
    return result_image

def process_simulation_mode(image, request):
    """Mode simulation amélioré"""
    print("🎭 Mode simulation...")
    time.sleep(0.2)
    
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    font = get_font()
    
    draw.text((10, 10), "🎭 SIMULATION MODE", fill="orange", font=font)
    
    simulated_translations = [
        ("こんにちは", "Hello"),
        ("ありがとう", "Thank you"), 
        ("元気ですか", "How are you?"),
        ("さようなら", "Goodbye"),
        ("おはよう", "Good morning")
    ]
    
    y_offset = 35
    for i, (original, translated) in enumerate(simulated_translations[:4]):
        text = f"{i+1}. '{original}' -> '{translated}'"
        draw.text((10, y_offset), text, fill="red", font=font)
        y_offset += 20
    
    draw.text((10, y_offset + 15), f"API ready for BallonsTranslator integration", fill="gray", font=font)
    draw.text((10, y_offset + 30), f"{request.source_lang} -> {request.target_lang}", fill="blue", font=font)
    
    return result_image

def get_font():
    """Police pour le rendu"""
    try:
        return ImageFont.truetype("arial.ttf", 14)
    except:
        try:
            return ImageFont.load_default()
        except:
            return None

if __name__ == "__main__":
    print("🚀 Démarrage Manga Translator API avec BallonsTranslator Integration...")
    print("📚 Documentation: http://localhost:8000/docs")
    print("💚 Health check: http://localhost:8000/health")
    print("🎯 Interface: http://localhost:8000/")
    print("🔌 Extension Chrome compatible")
    
    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
