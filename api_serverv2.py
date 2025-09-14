import os
import sys
import io
import time
import base64
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import uvicorn

# Qt + Canvas
from PySide6.QtWidgets import QApplication
from ui.canvas import Canvas

# Ajouter le chemin vers BallonsTranslator
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Modules BallonsTranslator
translator_ready = False
ballons_modules = {}
app_qt = None
canvas = None
qt_lock = threading.Lock()

# Importer les modules
try:
    print("🔍 Chargement des modules BallonsTranslator...")
    from modules.translators.trans_google import GoogleTranslator
    from modules.textdetector.detector_ctd import ComicTextDetector
    from modules.ocr.ocr_mit import OCRMIT48px
    from modules.inpaint.base import LamaLarge
    translator_ready = True
    print("✅ Modules importés avec succès")
except ImportError as e:
    print(f"⚠️ Import des modules échoué: {e}")
    translator_ready = False

# FastAPI lifespan
@asynccontextmanager
async def lifespan(app):
    global app_qt, canvas, ballons_modules

    print("🚀 Initialisation API Manga Translator...")
    print(f"📁 Répertoire de travail: {os.getcwd()}")

    print("🎨 Initialisation Qt + Canvas...")
    app_qt = QApplication([])
    canvas = Canvas()

    if translator_ready:
        print("🔧 Initialisation des modules BallonsTranslator...")
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
            print("✅ OCRMIT48px initialisé")
        except Exception as e:
            print(f"❌ Erreur OCR: {e}")

        try:
            ballons_modules['inpainter'] = LamaLarge()
            print("✅ LamaLarge initialisé")
        except Exception as e:
            print(f"❌ Erreur LamaLarge: {e}")

    yield

    print("🛑 Arrêt de l'API")
    if app_qt:
        app_qt.quit()

# FastAPI app
app = FastAPI(title="Manga Translator API",
              version="1.0.0",
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Pydantic models
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

# Endpoint santé
@app.get("/health")
async def health():
    status = "healthy" if translator_ready else "simulation_mode"
    print(f"🌡️  Health check: {status}")
    return {"status": status}

# Endpoint traduction
@app.post("/translate", response_model=TranslationResponse)
async def translate_image(request: TranslationRequest):
    start_time = time.time()
    try:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        print(f"📸 Image reçue: {image.width}x{image.height}")

        if translator_ready and ballons_modules:
            print("🎯 Workflow BallonsTranslator natif...")
            result_image = await translate_image_ballons_style(image, request)
        else:
            print("🎭 Mode simulation...")
            result_image = process_simulation_mode(image)

        buffer = io.BytesIO()
        result_image.save(buffer, format='PNG', optimize=True)
        result_base64 = base64.b64encode(buffer.getvalue()).decode()
        processing_time = time.time() - start_time
        print(f"✅ Traduction terminée en {processing_time:.2f}s")
        return TranslationResponse(success=True, translated_image_base64=result_base64, processing_time=processing_time)
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Erreur traduction: {e}")
        return TranslationResponse(success=False, error=str(e), processing_time=processing_time)

# Workflow BallonsTranslator
async def translate_image_ballons_style(image, request):
    try:
        img_array = np.array(image)
        detector = ballons_modules['detector']
        blk_list = []
        mask, blk_list = detector.detect(img_array, blk_list)
        print(f"📍 {len(blk_list)} blocs détectés")

        # OCR
        if 'ocr' in ballons_modules:
            ocr = ballons_modules['ocr']
            if hasattr(ocr, '_ocr_blk_list'):
                ocr._ocr_blk_list(img_array, blk_list)

        # Traduction
        if 'translator' in ballons_modules:
            translator = ballons_modules['translator']
            for blk in blk_list:
                text = blk.get_text()
                if text:
                    blk.translation = translator.translate(text, target_language=request.target_lang)
                    print(f"🔤 '{text}' -> '{blk.translation}'")

        # Inpainting + Canvas
        if 'inpainter' in ballons_modules and any(getattr(blk, 'translation', None) for blk in blk_list):
            inpainter = ballons_modules['inpainter']
            inpaint_mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
            for blk in blk_list:
                if getattr(blk, 'translation', None) and hasattr(blk, 'xyxy'):
                    x1, y1, x2, y2 = map(int, blk.xyxy)
                    inpaint_mask[y1:y2, x1:x2] = 255
            with qt_lock:
                qimage = canvas.render_result_img(img_array, blk_list, inpaint_mask)
            return qimage_to_pil(qimage)

        return render_ballons_overlay(image, blk_list)

    except Exception as e:
        print(f"❌ Erreur workflow: {e}")
        return image

# Helper functions
def render_ballons_overlay(image, blk_list):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    draw.text((10, 10), "🎯 BALLONS TRANSLATOR", fill="green", font=font)
    y = 35
    for i, blk in enumerate(blk_list[:6]):
        text = f"{i+1}. '{blk.get_text()}' -> '{getattr(blk, 'translation', '[no translation]')}'"
        draw.text((10, y), text, fill="white", font=font)
        y += 22
    return result

def process_simulation_mode(image):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    draw.text((10, 10), "🎭 SIMULATION MODE", fill="orange", font=ImageFont.load_default())
    return result

def qimage_to_pil(qimage):
    buffer = qimage.bits().asstring(qimage.width()*qimage.height()*4)
    img = Image.frombuffer("RGBA", (qimage.width(), qimage.height()), buffer, "raw", "BGRA", 0, 1)
    return img.convert("RGB")

# Lancer le serveur
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True, log_level="info")
