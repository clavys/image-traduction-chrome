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

# Import des modules BallonsTranslator
try:
    print("🔍 Chargement des modules BallonsTranslator...")
    
    from modules.translators.trans_google import GoogleTranslator
    from modules.textdetector.detector_ctd import ComicTextDetector  
    from modules.ocr.ocr_mit import OCRMIT48px
    from modules.inpaint.base import LamaLarge
    
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
            
            # Initialiser tous les modules
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
                ballons_modules['ocr'] = OCRMIT48px()
                # Charger le modèle OCR si nécessaire
                if hasattr(ballons_modules['ocr'], 'load_model'):
                    ballons_modules['ocr'].load_model()
                    print("✅ Modèle OCR chargé")
                print("✅ OCRMIT48px initialisé")
            except Exception as e:
                print(f"❌ Erreur OCR: {e}")
            
            try:
                ballons_modules['inpainter'] = LamaLarge()
                print("✅ LamaLarge initialisé")
            except Exception as e:
                print(f"❌ Erreur LamaLarge: {e}")
            
            modules_count = len(ballons_modules)
            print(f"🎯 {modules_count} modules initialisés: {list(ballons_modules.keys())}")
            
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
            print("🎯 Traitement avec workflow BallonsTranslator natif...")
            result_image = await translate_image_ballons_style(image, request)
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

# Fonction principale utilisant le workflow BallonsTranslator natif
async def translate_image_ballons_style(image, request):
    """Utiliser le workflow exact de BallonsTranslator comme dans scripts/run_module.py"""
    try:
        print("🔄 Workflow BallonsTranslator natif")
        
        # Conversion PIL -> numpy
        img_array = np.array(image)
        im_h, im_w = img_array.shape[:2]
        
        # 1. Détection des zones de texte (comme dans scripts/run_module.py)
        detector = ballons_modules['detector']
        blk_list = []  # Liste vide initiale comme dans le code source
        
        print("🔍 Détection des zones de texte...")
        mask, blk_list = detector.detect(img_array, blk_list)
        print(f"📍 {len(blk_list)} TextBlocks détectés")
        
        if not blk_list:
            return add_debug_info(image, "Aucune zone de texte détectée")
        
        # 2. OCR avec la vraie méthode interne (comme dans le code source)
        if 'ocr' in ballons_modules:
            ocr = ballons_modules['ocr']
            print("📖 OCR avec méthode interne BallonsTranslator...")
            
            try:
                # Utiliser la vraie méthode interne documentée dans le code source
                if hasattr(ocr, '_ocr_blk_list'):
                    print("    Utilisation de _ocr_blk_list (méthode interne)")
                    ocr._ocr_blk_list(img_array, blk_list)
                elif hasattr(ocr, 'run_ocr'):
                    print("    Utilisation de run_ocr")
                    result = ocr.run_ocr(img_array, blk_list)
                    if isinstance(result, list):
                        blk_list = result
                else:
                    print("    Méthode OCR non trouvée, utilisation manuelle")
                    # Fallback vers notre méthode manuelle
                    for blk in blk_list:
                        if hasattr(blk, 'xyxy'):
                            x1, y1, x2, y2 = map(int, blk.xyxy)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(im_w, x2), min(im_h, y2)
                            
                            if x2 > x1 and y2 > y1:
                                region_crop = img_array[y1:y2, x1:x2]
                                try:
                                    text = ocr.ocr_img(region_crop)
                                    if text and text.strip():
                                        blk.text = [text.strip()]
                                except Exception as e:
                                    print(f"    OCR manuel échoué: {e}")
                
            except Exception as e:
                print(f"❌ Erreur OCR interne: {e}")
        
        # 3. Traduction (comme dans le workflow BallonsTranslator)
        if 'translator' in ballons_modules:
            translator = ballons_modules['translator']
            print("🌐 Traduction des TextBlocks...")
            
            translated_count = 0
            for blk in blk_list:
                text = blk.get_text()
                if text and text.strip():
                    try:
                        # Utiliser l'API du traducteur BallonsTranslator
                        translation = translator.translate(text, target_language=request.target_lang)
                        blk.translation = translation
                        translated_count += 1
                        print(f"🔄 '{text}' -> '{translation}'")
                    except Exception as e:
                        print(f"⚠️ Erreur traduction: {e}")
                        blk.translation = f"[ERREUR] {text}"
            
            print(f"📝 {translated_count} blocs traduits")
        
        # 4. Inpainting et rendu final (comme dans BallonsTranslator)
        if 'inpainter' in ballons_modules and any(blk.translation for blk in blk_list):
            print("🖌️ Inpainting et rendu final...")
            return render_ballons_result(image, img_array, blk_list, mask)
        else:
            return render_ballons_overlay(image, blk_list)
        
    except Exception as e:
        print(f"❌ Erreur workflow BallonsTranslator: {e}")
        import traceback
        traceback.print_exc()
        return add_debug_info(image, f"Erreur workflow: {str(e)}")

def render_ballons_result(original_image, img_array, blk_list, mask):
    """Rendu final avec inpainting utilisant les propriétés BallonsTranslator"""
    try:
        inpainter = ballons_modules['inpainter']
        
        # Créer le masque pour l'inpainting
        inpaint_mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
        
        # Utiliser les zones détectées pour créer le masque
        for blk in blk_list:
            if blk.translation and hasattr(blk, 'xyxy'):
                x1, y1, x2, y2 = map(int, blk.xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(inpaint_mask.shape[1], x2), min(inpaint_mask.shape[0], y2)
                inpaint_mask[y1:y2, x1:x2] = 255
        
        # Appliquer l'inpainting
        try:
            inpainted_array = inpainter.inpaint(img_array, inpaint_mask)
            result_image = Image.fromarray(inpainted_array.astype(np.uint8))
        except Exception as e:
            print(f"⚠️ Inpainting échoué: {e}")
            result_image = original_image.copy()
        
        # Rendu du texte utilisant les propriétés BallonsTranslator
        draw = ImageDraw.Draw(result_image)
        
        for blk in blk_list:
            if blk.translation and hasattr(blk, 'xyxy'):
                x1, y1, x2, y2 = map(int, blk.xyxy)
                zone_width = x2 - x1
                zone_height = y2 - y1
                
                # Utiliser la taille de police détectée par BallonsTranslator
                detected_font_size = getattr(blk, 'font_size', 16)
                if hasattr(blk, 'fontformat') and blk.fontformat.font_size > 0:
                    detected_font_size = int(blk.fontformat.font_size)
                elif hasattr(blk, '_detected_font_size') and blk._detected_font_size > 0:
                    detected_font_size = int(blk._detected_font_size)
                
                # Adapter la taille à la zone disponible
                font_size = min(detected_font_size, zone_height // 2, zone_width // 8)
                font_size = max(font_size, 8)  # Minimum 8px
                
                # Utiliser les propriétés de formatage de BallonsTranslator
                vertical = getattr(blk, 'vertical', False)
                alignment = getattr(blk, 'alignment', 1)  # 1 = center par défaut
                
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                
                # Gérer l'orientation et la découpe du texte
                translation = blk.translation
                if vertical:
                    # Pour le texte vertical, limiter par la hauteur
                    max_chars_per_line = max(1, zone_height // font_size)
                    lines = [translation[i:i+max_chars_per_line] for i in range(0, len(translation), max_chars_per_line)]
                else:
                    # Pour le texte horizontal, découper par mots
                    words = translation.split()
                    lines = []
                    current_line = ""
                    
                    for word in words:
                        test_line = current_line + (" " if current_line else "") + word
                        text_width = draw.textlength(test_line, font=font)
                        
                        if text_width <= zone_width - 10:
                            current_line = test_line
                        else:
                            if current_line:
                                lines.append(current_line)
                                current_line = word
                            else:
                                lines.append(word[:zone_width // (font_size // 2)])
                    
                    if current_line:
                        lines.append(current_line)
                
                # Limiter le nombre de lignes selon la zone
                max_lines = max(1, zone_height // (font_size + 4))
                if len(lines) > max_lines:
                    lines = lines[:max_lines-1] + [lines[max_lines-1][:8] + "..."]
                
                # Calculer le positionnement selon l'alignement BallonsTranslator
                total_text_height = len(lines) * (font_size + 4)
                
                if vertical:
                    # Texte vertical (manga japonais)
                    start_x = x1 + zone_width - font_size - 5
                    start_y = y1 + (zone_height - total_text_height) // 2
                else:
                    # Texte horizontal
                    start_y = y1 + (zone_height - total_text_height) // 2
                
                # Dessiner le texte ligne par ligne
                for i, line in enumerate(lines):
                    if vertical:
                        text_x = start_x
                        text_y = start_y + i * (font_size + 4)
                    else:
                        # Alignement horizontal selon BallonsTranslator
                        text_width = draw.textlength(line, font=font)
                        if alignment == 0:  # Left
                            text_x = x1 + 5
                        elif alignment == 2:  # Right
                            text_x = x2 - text_width - 5
                        else:  # Center (défaut)
                            text_x = x1 + (zone_width - text_width) // 2
                        
                        text_y = start_y + i * (font_size + 4)
                    
                    # Vérifier les limites de l'image
                    text_x = max(2, min(text_x, result_image.width - 2))
                    text_y = max(2, min(text_y, result_image.height - font_size - 2))
                    
                    if text_y + font_size > result_image.height:
                        break
                    
                    # Fond blanc avec opacité pour lisibilité
                    text_bbox = draw.textbbox((text_x, text_y), line, font=font)
                    padding = 2
                    bg_bbox = [
                        text_bbox[0] - padding,
                        text_bbox[1] - padding,
                        text_bbox[2] + padding,
                        text_bbox[3] + padding
                    ]
                    
                    # Créer un fond blanc semi-transparent
                    overlay = Image.new('RGBA', result_image.size, (255, 255, 255, 0))
                    overlay_draw = ImageDraw.Draw(overlay)
                    overlay_draw.rectangle(bg_bbox, fill=(255, 255, 255, 200))
                    
                    # Composer avec l'image principale
                    result_image = Image.alpha_composite(
                        result_image.convert('RGBA'), 
                        overlay
                    ).convert('RGB')
                    
                    # Redessiner sur l'image composée
                    draw = ImageDraw.Draw(result_image)
                    
                    # Texte noir
                    draw.text((text_x, text_y), line, fill="black", font=font)
        
        # Signature discrète
        small_font = get_font(size=8)
        draw.text((5, result_image.height - 12), "BallonsTranslator", fill=(100, 100, 100), font=small_font)
        
        return result_image
        
    except Exception as e:
        print(f"❌ Erreur rendu final: {e}")
        return render_ballons_overlay(original_image, blk_list)

def render_ballons_overlay(image, blk_list):
    """Rendu simple avec overlay des traductions"""
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    font = get_font()
    
    # Titre
    draw.text((10, 10), "🎯 BALLONS TRANSLATOR - DONNÉES RÉELLES", fill="green", font=font)
    
    y_offset = 35
    for i, blk in enumerate(blk_list[:6]):
        original_text = blk.get_text()
        translated_text = getattr(blk, 'translation', '[pas de traduction]')
        
        if original_text:
            text = f"{i+1}. '{original_text}' -> '{translated_text}'"
            
            # Fond semi-transparent
            text_bbox = draw.textbbox((10, y_offset), text, font=font)
            draw.rectangle([(text_bbox[0]-2, text_bbox[1]-2), (text_bbox[2]+2, text_bbox[3]+2)], 
                          fill=(0, 0, 0, 180))
            
            # Texte
            draw.text((10, y_offset), text, fill="white", font=font)
            y_offset += 22
    
    if len(blk_list) > 6:
        draw.text((10, y_offset + 5), f"... et {len(blk_list)-6} autres zones", fill="gray", font=font)
    
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

def get_font(size=14):
    """Police pour le rendu avec taille personnalisable"""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        try:
            # Essayer d'autres polices système
            for font_name in ["calibri.ttf", "tahoma.ttf", "verdana.ttf"]:
                try:
                    return ImageFont.truetype(font_name, size)
                except:
                    continue
            return ImageFont.load_default()
        except:
            return None

if __name__ == "__main__":
    print("🚀 Démarrage Manga Translator API avec BallonsTranslator Workflow Natif...")
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
