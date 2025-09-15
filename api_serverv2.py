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
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

thread_pools = {
    'ocr': ThreadPoolExecutor(max_workers=8, thread_name_prefix="OCR"),
    'translation': ThreadPoolExecutor(max_workers=4, thread_name_prefix="TRANSLATE"),
    'inpaint': ThreadPoolExecutor(max_workers=2, thread_name_prefix="INPAINT") 
}

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
            print("OCR multi-threadé avec _ocr_blk_list...")
            
            # Préparer les tâches
            loop = asyncio.get_event_loop()
            tasks = []
            
            for i, blk in enumerate(blk_list):
                if hasattr(blk, 'xyxy'):
                    task = loop.run_in_executor(
                        thread_pools['ocr'],
                        ocr_single_block_with_ballons,
                        img_array, blk, i
                    )
                    tasks.append(task)
            
            # Exécuter en parallèle
            if tasks:
                print(f"Lancement de {len(tasks)} threads OCR...")
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Appliquer les résultats
                success_count = 0
                for result in results:
                    if not isinstance(result, Exception):
                        blk_index, text = result
                        if blk_index < len(blk_list) and text:
                            blk_list[blk_index].text = [text]
                            success_count += 1
                            print(f"OCR thread {blk_index}: '{text}'")
                
                print(f"OCR terminé: {success_count}/{len(tasks)} réussis")
        

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

def get_font(size=18):
    """Police pour le rendu avec meilleure lisibilité"""
    try:
        # Essayer DejaVuSans-Bold, souvent présente avec Pillow
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except:
        try:
            return ImageFont.load_default()
        except:
            return None


def draw_text_with_outline(draw, position, text, font, fill="white", outline="black", stroke_width=2):
    """Dessine du texte avec contour"""
    draw.text(position, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=outline)


def render_ballons_result(original_image, img_array, blk_list, mask):
    """Rendu final avec texte blanc contour noir"""
    try:
        inpainter = ballons_modules.get('inpainter')
        inpaint_mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
        for blk in blk_list:
            if blk.translation and hasattr(blk, 'xyxy'):
                x1, y1, x2, y2 = map(int, blk.xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(inpaint_mask.shape[1], x2), min(inpaint_mask.shape[0], y2)
                inpaint_mask[y1:y2, x1:x2] = 255

        try:
            inpainted_array = inpainter.inpaint(img_array, inpaint_mask)
            result_image = Image.fromarray(inpainted_array.astype(np.uint8))
        except Exception as e:
            print(f"⚠️ Inpainting échoué: {e}")
            result_image = original_image.copy()

        draw = ImageDraw.Draw(result_image)
        font = get_font(18)
        ascent, descent = font.getmetrics()
        line_spacing = ascent + descent + 2

        for blk in blk_list:
            if blk.translation and hasattr(blk, 'xyxy'):
                x1, y1, x2, y2 = map(int, blk.xyxy)
                max_width = x2 - x1
                max_height = y2 - y1
                lines = wrap_text(blk.translation, font, max_width, draw)
                total_height = len(lines) * line_spacing
                y_text = y1 + (max_height - total_height) // 2

                for line in lines:
                    w = draw.textlength(line, font=font)
                    x_text = x1 + (max_width - w) // 2
                    draw_text_with_outline(draw, (x_text, y_text), line, font)
                    y_text += line_spacing

        draw_text_with_outline(draw, (10, original_image.height - 25),
                               "🎯 BALLONS TRANSLATOR - WORKFLOW NATIF", font, fill="green", outline="black", stroke_width=2)
        return result_image

    except Exception as e:
        print(f"❌ Erreur rendu final: {e}")
        return render_ballons_overlay(original_image, blk_list)


def render_ballons_overlay(image, blk_list):
    """Rendu simple avec texte blanc contour noir"""
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    font = get_font(16)
    ascent, descent = font.getmetrics()
    line_height = ascent + descent + 2

    draw_text_with_outline(draw, (10, 10), "🎯 BALLONS TRANSLATOR - DONNÉES RÉELLES", font, fill="green", outline="black")

    y_offset = 35
    for i, blk in enumerate(blk_list[:6]):
        original_text = blk.get_text()
        translated_text = getattr(blk, 'translation', '[pas de traduction]')
        if original_text:
            text = f"{i+1}. '{original_text}' -> '{translated_text}'"
            draw_text_with_outline(draw, (10, y_offset), text, font)
            y_offset += line_height

    if len(blk_list) > 6:
        draw_text_with_outline(draw, (10, y_offset + 5), f"... et {len(blk_list)-6} autres zones", font, fill="gray")

    return result_image


def wrap_text(text, font, max_width, draw):
    """Découper le texte en plusieurs lignes pour tenir dans max_width"""
    words = text.split(' ')
    lines = []
    current_line = ''
    for word in words:
        test_line = f"{current_line} {word}".strip()
        w = draw.textlength(test_line, font=font)
        if w <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

"""multi-threading OCR"""
def ocr_single_block_with_ballons(img_array, blk, blk_index):
    """OCR d'un bloc avec la méthode BallonsTranslator"""
    try:
        # Créer une liste temporaire avec juste ce bloc
        temp_blk_list = [blk]
        
        ocr = ballons_modules['ocr']
        if hasattr(ocr, '_ocr_blk_list'):
            print(f"    Thread {blk_index}: Utilisation de _ocr_blk_list")
            ocr._ocr_blk_list(img_array, temp_blk_list)
            
            # Récupérer le texte du bloc traité
            if temp_blk_list and hasattr(temp_blk_list[0], 'text'):
                text = temp_blk_list[0].text
                if isinstance(text, list):
                    text = " ".join(text) if text else ""
                return blk_index, str(text).strip()
        
        return blk_index, ""
        
    except Exception as e:
        print(f"Erreur OCR thread {blk_index}: {e}")
        return blk_index, ""

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
