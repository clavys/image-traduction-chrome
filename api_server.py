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
ballons_translator_path = r"C:\Users\reppe\BallonsTranslator"
sys.path.insert(0, ballons_translator_path)

# Aussi ajouter le répertoire courant (où est api_server.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print(f"📁 Chemin BallonsTranslator: {ballons_translator_path}")
print(f"📁 Répertoire courant: {current_dir}")

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
    translator_ready = False

# Configuration FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    global translator_ready, ballons_modules
    
    if translator_ready:
        print("🔧 Initialisation des modules BallonsTranslator...")
        
        try:
            ballons_modules['translator'] = GoogleTranslator()
            ballons_modules['detector'] = ComicTextDetector()
            ballons_modules['ocr'] = OCRMIT48px()
            ballons_modules['inpainter'] = LamaLarge()
            
            print(f"🎯 {len(ballons_modules)} modules initialisés")
            print("🎊 BallonsTranslator prêt!")
            
        except Exception as e:
            print(f"❌ Erreur initialisation: {e}")
            translator_ready = False
    
    yield

# Créer l'app FastAPI
app = FastAPI(title="Manga Translator API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles
class TranslationRequest(BaseModel):
    image_base64: str
    source_lang: str = "ja"
    target_lang: str = "en"

class TranslationResponse(BaseModel):
    success: bool
    translated_image_base64: str = None
    error: str = None
    processing_time: float = 0

@app.get("/")
async def root():
    return {
        "name": "Manga Translator API",
        "status": "running",
        "ballons_translator": translator_ready,
        "modules": list(ballons_modules.keys()) if translator_ready else []
    }

@app.post("/translate", response_model=TranslationResponse)
async def translate_image(request: TranslationRequest):
    start_time = time.time()
    
    try:
        # Décoder l'image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        print(f"📸 Image reçue: {image.width}x{image.height}")
        
        # Traitement
        if translator_ready:
            result_image = await process_with_ballons(image, request)
        else:
            result_image = create_error_image(image, "BallonsTranslator non disponible")
        
        # Encoder résultat
        buffer = io.BytesIO()
        result_image.save(buffer, format='PNG')
        result_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        processing_time = time.time() - start_time
        print(f"✅ Terminé en {processing_time:.2f}s")
        
        return TranslationResponse(
            success=True,
            translated_image_base64=result_base64,
            processing_time=processing_time
        )
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        processing_time = time.time() - start_time
        
        return TranslationResponse(
            success=False,
            error=str(e),
            processing_time=processing_time
        )

@app.post("/translate-file")
async def translate_file(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Fichier doit être une image")
    
    image_data = await file.read()
    image_base64 = base64.b64encode(image_data).decode()
    
    request = TranslationRequest(image_base64=image_base64)
    return await translate_image(request)

# ==================================================================================
# TRAITEMENT PRINCIPAL AVEC BALLONSTRANSLATOR
# ==================================================================================

async def process_with_ballons(image, request):
    """Traitement principal avec BallonsTranslator"""
    try:
        print("🔄 Traitement avec BallonsTranslator...")
        
        img_array = np.array(image)
        
        # 1. Détection
        detector = ballons_modules['detector']
        mask, blk_list = detector.detect(img_array, [])
        print(f"📍 {len(blk_list)} zones détectées")
        
        if not blk_list:
            return create_error_image(image, "Aucune zone de texte détectée")
        
        # 2. OCR - Version qui fonctionnait avant
        ocr = ballons_modules['ocr']
        print("📖 OCR avec méthode qui marchait...")
        
        # Utiliser la méthode manuelle qui fonctionnait
        for blk in blk_list:
            if hasattr(blk, 'xyxy'):
                x1, y1, x2, y2 = map(int, blk.xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_array.shape[1], x2), min(img_array.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    region_crop = img_array[y1:y2, x1:x2]
                    try:
                        text = ocr.ocr_img(region_crop)
                        if text and text.strip():
                            blk.text = [text.strip()]
                            print(f"📖 Texte détecté: '{text.strip()}'")
                    except Exception as e:
                        print(f"❌ OCR région échoué: {e}")
                        blk.text = [""]
        
        print("📖 OCR terminé")
        
        # 3. Traduction avec la méthode qui marchait
        translator = ballons_modules['translator']
        translated_count = 0
        
        for blk in blk_list:
            try:
                # Récupérer le texte de la façon qui marchait
                if hasattr(blk, 'text') and blk.text and blk.text[0].strip():
                    text = blk.text[0].strip()
                    
                    translation = translator.translate(text, target_language=request.target_lang)
                    blk.translation = translation
                    translated_count += 1
                    print(f"🔄 '{text}' -> '{translation}'")
                    
            except Exception as e:
                print(f"⚠️ Erreur sur un bloc: {e}")
                continue
        
        print(f"📝 {translated_count} blocs traduits")
        
        if translated_count == 0:
            return create_error_image(image, "Aucune traduction réussie")
        
        # 4. Rendu avec BallonsTranslator (uniquement si on a des traductions)
        try:
            result_img = render_with_ballons_native(img_array, blk_list)
            
            if result_img is not None:
                return Image.fromarray(result_img.astype(np.uint8))
            else:
                print("⚠️ Rendu natif échoué, utilisation du rendu simple")
                return render_simple(image, blk_list)
                
        except Exception as render_error:
            print(f"❌ Erreur rendu natif: {render_error}")
            print("🎨 Utilisation du rendu simple de secours")
            return render_simple(image, blk_list)
        
    except Exception as e:
        print(f"❌ Erreur traitement: {e}")
        return create_error_image(image, str(e))

# ==================================================================================
# RENDU AVEC BALLONSTRANSLATOR NATIF
# ==================================================================================

def render_with_ballons_native(img_array, blk_list):
    """Essaie d'utiliser les méthodes natives de BallonsTranslator"""
    
    # Méthode 1: ModuleManager (workflow complet)
    try:
        from ui.module_manager import ModuleManager
        
        manager = ModuleManager()
        manager.detector = ballons_modules.get('detector')
        manager.ocr = ballons_modules.get('ocr')
        manager.translator = ballons_modules.get('translator')
        manager.inpainter = ballons_modules.get('inpainter')
        
        result = manager.run_translation_pipeline(
            img_array, 
            textblocks=blk_list,
            skip_detection=True,
            skip_ocr=True,
            skip_translation=True,
            inpaint=True,
            render_text=True
        )
        
        print("✅ Rendu avec ModuleManager natif")
        return result
        
    except Exception as e:
        print(f"⚠️ ModuleManager échoué: {e}")
    
    # Méthode 2: Canvas export
    try:
        from ui.canvas import Canvas
        from utils.imgproc_utils import ndarray2qimage, qimage2ndarray
        
        canvas = Canvas()
        canvas.load_image(ndarray2qimage(img_array))
        canvas.textblk_lst = blk_list
        canvas.inpainter = ballons_modules.get('inpainter')
        
        result_qimg = canvas.export_rendered_image()
        result_array = qimage2ndarray(result_qimg)
        
        print("✅ Rendu avec Canvas export")
        return result_array
        
    except Exception as e:
        print(f"⚠️ Canvas export échoué: {e}")
    
    # Méthode 3: Inpainting seul
    try:
        inpainter = ballons_modules.get('inpainter')
        if inpainter:
            # Créer masque des zones de texte
            mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
            for blk in blk_list:
                if hasattr(blk, 'xyxy'):
                    x1, y1, x2, y2 = map(int, blk.xyxy)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_array.shape[1], x2), min(img_array.shape[0], y2)
                    mask[y1:y2, x1:x2] = 255
            
            result = inpainter.inpaint(img_array, mask)
            print("✅ Inpainting natif appliqué")
            return result
            
    except Exception as e:
        print(f"⚠️ Inpainting échoué: {e}")
    
    print("❌ Toutes les méthodes natives ont échoué")
    return None

# ==================================================================================
# RENDU SIMPLE DE SECOURS
# ==================================================================================

def render_simple(image, blk_list):
    """Rendu simple et efficace pour l'anglais"""
    print("🎨 Rendu simple de secours")
    
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    
    for blk in blk_list:
        if not blk.translation or not hasattr(blk, 'xyxy'):
            continue
            
        x1, y1, x2, y2 = map(int, blk.xyxy)
        zone_width = x2 - x1
        zone_height = y2 - y1
        
        # Police adaptée
        font_size = min(zone_height // 4, zone_width // 10, 24)
        font_size = max(font_size, 12)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Découper le texte en lignes (HORIZONTAL uniquement)
        words = blk.translation.split()
        lines = []
        current_line = ""
        margin = 15
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            text_width = draw.textlength(test_line, font=font)
            
            if text_width <= zone_width - margin:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    lines.append(word[:15] + "..." if len(word) > 15 else word)
        
        if current_line:
            lines.append(current_line)
        
        # Limiter les lignes
        max_lines = max(1, zone_height // (font_size + 5))
        lines = lines[:max_lines]
        
        # Positionner le texte
        line_height = font_size + 5
        total_height = len(lines) * line_height
        start_y = y1 + (zone_height - total_height) // 2
        
        # Fond blanc pour lisibilité
        draw.rectangle([x1, start_y - 3, x2, start_y + total_height + 3], fill="white", outline="lightgray")
        
        # Dessiner le texte
        for i, line in enumerate(lines):
            text_width = draw.textlength(line, font=font)
            text_x = x1 + (zone_width - text_width) // 2
            text_y = start_y + i * line_height
            
            # Ombre légère
            draw.text((text_x + 1, text_y + 1), line, fill="gray", font=font)
            # Texte principal
            draw.text((text_x, text_y), line, fill="black", font=font)
    
    return result_image

def create_error_image(image, message):
    """Créer une image d'erreur"""
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    draw.rectangle([10, 10, 400, 60], fill="red")
    draw.text((15, 15), f"Erreur: {message}", fill="white", font=font)
    
    return result_image

# ==================================================================================
# DÉMARRAGE
# ==================================================================================

if __name__ == "__main__":
    print("🚀 Démarrage Manga Translator API Simple...")
    print("📚 Documentation: http://localhost:8000/docs")
    print("🎯 Interface: http://localhost:8000/")
    
    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
