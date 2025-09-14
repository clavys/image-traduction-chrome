# api_server.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64, io, os, sys, time
from PIL import Image
import numpy as np
import uvicorn

# Qt & BallonsTranslator imports
from PyQt5.QtWidgets import QApplication
from ui.canvas import Canvas

# Ajouter le chemin vers BallonsTranslator
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

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
    print(f"⚠️ Import échoué: {e}. Mode simulation activé")
    translator_ready = False

# Lifespan FastAPI
from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app):
    global global translator_ready, ballons_modules
    try:
        print("🚀 Initialisation API Manga Translator...")
        print(f"📁 Répertoire de travail: {os.getcwd()}")
        if translator_ready:
            ballons_modules = {}
            try: ballons_modules['translator'] = GoogleTranslator(); print("✅ GoogleTranslator initialisé")
            except Exception as e: print(f"❌ GoogleTranslator: {e}")
            try: ballons_modules['detector'] = ComicTextDetector(); print("✅ ComicTextDetector initialisé")
            except Exception as e: print(f"❌ ComicTextDetector: {e}")
            try:
                ballons_modules['ocr'] = OCRMIT48px()
                if hasattr(ballons_modules['ocr'], 'load_model'): ballons_modules['ocr'].load_model()
                print("✅ OCRMIT48px initialisé")
            except Exception as e: print(f"❌ OCR: {e}")
            try: ballons_modules['inpainter'] = LamaLarge(); print("✅ LamaLarge initialisé")
            except Exception as e: print(f"❌ LamaLarge: {e}")
        print("🎯 API prête!")
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        translator_ready = False
    yield
    print("🛑 Arrêt de l'API")

app = FastAPI(title="Manga Translator API - BallonsTranslator", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Modèles
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

# Qt App global
_qt_app = None
def get_qt_app():
    global _qt_app
    if not _qt_app: _qt_app = QApplication([])
    return _qt_app

# Routes
@app.get("/")
async def root():
    return {
        "name": "Manga Translator API",
        "version": "1.0.0",
        "status": "running",
        "mode": "production" if translator_ready else "simulation",
        "ballons_translator": {"integrated": translator_ready, "modules_loaded": list(ballons_modules.keys()) if translator_ready else [], "status": "✅ Operational" if translator_ready else "⚠️ Simulation Mode"},
        "endpoints": {"translate": "POST /translate", "translate_file": "POST /translate-file", "health": "GET /health", "docs": "GET /docs"},
        "chrome_extension": {"compatible": True, "cors_enabled": True}
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if translator_ready else "simulation_mode",
        "modules": list(ballons_modules.keys()) if translator_ready else [],
        "message": "🎊 BallonsTranslator fully integrated!" if translator_ready else "🎭 Running in simulation mode"
    }

@app.post("/translate", response_model=TranslationResponse)
async def translate_image(request: TranslationRequest):
    start_time = time.time()
    try:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        print(f"📸 Image reçue: {image.width}x{image.height}")

        if translator_ready and ballons_modules:
            result_image = await translate_ballons_workflow(image, request)
        else:
            result_image = simulation_render(image, request)

        buf = io.BytesIO()
        result_image.save(buf, format='PNG', optimize=True)
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        return TranslationResponse(success=True, translated_image_base64=img_base64, processing_time=time.time()-start_time)

    except Exception as e:
        return TranslationResponse(success=False, error=str(e), processing_time=time.time()-start_time)

@app.post("/translate-file")
async def translate_file(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Fichier doit être une image")
    data = await file.read()
    req = TranslationRequest(image_base64=base64.b64encode(data).decode())
    return await translate_image(req)

# Functions
async def translate_ballons_workflow(image, request):
    img_array = np.array(image)
    detector = ballons_modules['detector']
    blk_list = []
    mask, blk_list = detector.detect(img_array, blk_list)

    # OCR
    if 'ocr' in ballons_modules:
        ocr = ballons_modules['ocr']
        if hasattr(ocr, '_ocr_blk_list'): ocr._ocr_blk_list(img_array, blk_list)
        elif hasattr(ocr, 'run_ocr'): blk_list = ocr.run_ocr(img_array, blk_list)

    # Traduction
    if 'translator' in ballons_modules:
        translator = ballons_modules['translator']
        for blk in blk_list:
            text = blk.get_text()
            if text: blk.translation = translator.translate(text, target_language=request.target_lang)

    # Rendu natif via Canvas
    if 'inpainter' in ballons_modules:
        return render_ballons_native(image, blk_list)
    else:
        return render_overlay(image, blk_list)

def render_ballons_native(image_pil, blk_list):
    try:
        get_qt_app()
        canvas = Canvas()
        canvas.set_image(np.array(image_pil))
        canvas.set_blocks(blk_list)
        rendered_qt = canvas.render_result_img()
        if hasattr(rendered_qt, 'bits'):  # QImage
            width, height = rendered_qt.width(), rendered_qt.height()
            ptr = rendered_qt.bits(); ptr.setsize(rendered_qt.byteCount())
            arr = np.array(ptr).reshape(height, width, 4)
            return Image.fromarray(arr[..., :3])
        else:
            return Image.fromarray(rendered_qt)
    except Exception as e:
        print(f"❌ Erreur Canvas natif: {e}")
        return render_overlay(image_pil, blk_list)

def render_overlay(image, blk_list):
    result = image.copy()
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    y_offset = 10
    for i, blk in enumerate(blk_list[:6]):
        text = f"{i+1}. {getattr(blk, 'translation', '[?]')}"
        draw.text((10, y_offset), text, fill="red", font=font)
        y_offset += 20
    return result

def simulation_render(image, request):
    result = image.copy()
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    draw.text((10,10),"🎭 SIMULATION MODE",fill="orange",font=font)
    return result

if __name__ == "__main__":
    print("🚀 Démarrage Manga Translator API avec BallonsTranslator Workflow Natif...")
    print("📚 Documentation: http://localhost:8000/docs")
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True, log_level="info")
