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

# Qt (BallonsTranslator UI rendering)
from PySide6.QtWidgets import QApplication
from ui.canvas import Canvas
from PySide6.QtGui import QImage

# Ajouter le chemin vers BallonsTranslator
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# État des modules BallonsTranslator
translator_ready = False
ballons_modules = {}
app_qt = None
canvas = None

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
    global translator_ready, ballons_modules, app_qt, canvas
    
    try:
        print("🚀 Initialisation de l'API Manga Translator...")
        print(f"📁 Répertoire de travail: {os.getcwd()}")

        # Init QApplication et Canvas une seule fois
        print("🎨 Initialisation Qt + Canvas...")
        app_qt = QApplication([])
        canvas = Canvas()

        if translator_ready:
            print("🔧 Initialisation des modules BallonsTranslator...")
            
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
                ballons_modules['ocr'] = OCRMIT48px()
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
    
    print("🛑 Arrêt de l'API")
    if app_qt:
        app_qt.quit()

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
            "translate": "POST /translate",
            "translate_file": "POST /translate-file",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }

@app.post("/translate", response_model=TranslationResponse)
async def translate_image(request: TranslationRequest):
    start_time = time.time()
    
    try:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        print(f"📸 Image reçue: {image.width}x{image.height}")

        if translator_ready and ballons_modules:
            print("🎯 Workflow BallonsTranslator natif...")
            result_image = await translate_image_ballons_style(image, request)
        else:
            print("🎭 Mode simulation...")
            result_image = process_simulation_mode(image, request)
        
        buffer = io.BytesIO()
        result_image.save(buffer, format='PNG', optimize=True)
        result_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        processing_time = time.time() - start_time
        print(f"✅ Terminé en {processing_time:.2f}s")
        
        return TranslationResponse(
            success=True,
            translated_image_base64=result_base64,
            processing_time=processing_time
        )
    except Exception as e:
        processing_time = time.time() - start_time
        return TranslationResponse(success=False, error=str(e), processing_time=processing_time)

# Fonction principale avec workflow BallonsTranslator
async def translate_image_ballons_style(image, request):
    try:
        img_array = np.array(image)
        detector = ballons_modules['detector']
        blk_list = []
        mask, blk_list = detector.detect(img_array, blk_list)
        print(f"📍 {len(blk_list)} TextBlocks détectés")

        if 'ocr' in ballons_modules:
            ocr = ballons_modules['ocr']
            if hasattr(ocr, '_ocr_blk_list'):
                ocr._ocr_blk_list(img_array, blk_list)

        if 'translator' in ballons_modules:
            translator = ballons_modules['translator']
            for blk in blk_list:
                text = blk.get_text()
                if text:
                    blk.translation = translator.translate(text, target_language=request.target_lang)

        if 'inpainter' in ballons_modules and any(blk.translation for blk in blk_list):
            inpainter = ballons_modules['inpainter']
            inpaint_mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
            for blk in blk_list:
                if blk.translation and hasattr(blk, 'xyxy'):
                    x1, y1, x2, y2 = map(int, blk.xyxy)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(inpaint_mask.shape[1], x2), min(inpaint_mask.shape[0], y2)
                    inpaint_mask[y1:y2, x1:x2] = 255

            inpainted_array = inpainter.inpaint(img_array, inpaint_mask)

            # 👉 Rendu avec Canvas Qt
            qimage = canvas.render_result_img(inpainted_array, blk_list, inpaint_mask)
            return qimage_to_pil(qimage)

        return render_ballons_overlay(image, blk_list)

    except Exception as e:
        print(f"❌ Erreur workflow: {e}")
        return image

def render_ballons_overlay(image, blk_list):
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    font = get_font()
    draw.text((10, 10), "🎯 BALLONS TRANSLATOR - DONNÉES RÉELLES", fill="green", font=font)
    y_offset = 35
    for i, blk in enumerate(blk_list[:6]):
        text = f"{i+1}. '{blk.get_text()}' -> '{getattr(blk, 'translation', '[no translation]')}'"
        draw.text((10, y_offset), text, fill="white", font=font)
        y_offset += 22
    return result_image

def process_simulation_mode(image, request):
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    font = get_font()
    draw.text((10, 10), "🎭 SIMULATION MODE", fill="orange", font=font)
    return result_image

def get_font():
    try:
        return ImageFont.truetype("arial.ttf", 14)
    except:
        return ImageFont.load_default()

def qimage_to_pil(qimage: QImage):
    buffer = qimage.bits().asstring(qimage.width() * qimage.height() * 4)
    img = Image.frombuffer("RGBA", (qimage.width(), qimage.height()), buffer, "raw", "BGRA", 0, 1)
    return img.convert("RGB")

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True, log_level="info")
