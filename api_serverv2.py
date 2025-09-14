from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64, io, time
from PIL import Image, ImageDraw, ImageFont
import uvicorn
import numpy as np
import os, sys

# Qt (Canvas rendering)
from PySide6.QtWidgets import QApplication
from ui.canvas import Canvas
from PySide6.QtGui import QImage

# Globals
app_qt = None
canvas = None

# Dummy modules (pour simulation si BallonsTranslator n’est pas installé)
translator_ready = False
ballons_modules = {}

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

# FastAPI avec lifespan pour Qt
from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app):
    global app_qt, canvas
    app_qt = QApplication([])  # QApplication unique
    canvas = Canvas()          # Canvas partagé
    yield
    app_qt.quit()

# App FastAPI
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints
@app.get("/")
async def root():
    return {"status": "running", "mode": "production" if translator_ready else "simulation"}

@app.post("/translate", response_model=TranslationResponse)
async def translate_image(request: TranslationRequest):
    start_time = time.time()
    try:
        # Decode base64
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Simulation / BallonsTranslator workflow
        result_image = process_image_with_canvas(image)
        
        # Encode result
        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG", optimize=True)
        result_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        processing_time = time.time() - start_time
        return TranslationResponse(
            success=True,
            translated_image_base64=result_base64,
            processing_time=processing_time
        )
    except Exception as e:
        return TranslationResponse(success=False, error=str(e), processing_time=time.time()-start_time)

# Fonction qui utilise Canvas Qt pour rendu
def process_image_with_canvas(image):
    # Convert PIL -> np array
    img_array = np.array(image)
    # Dummy text blocks
    blk_list = [{"text": "Hello", "translation": "Bonjour", "xyxy": [50,50,200,100]}]

    # Utilisation de Canvas Qt
    qimage = canvas.render_result_img(img_array, blk_list, mask=None)
    return qimage_to_pil(qimage)

def qimage_to_pil(qimage: QImage):
    buffer = qimage.bits().asstring(qimage.width() * qimage.height() * 4)
    img = Image.frombuffer("RGBA", (qimage.width(), qimage.height()), buffer, "raw", "BGRA", 0, 1)
    return img.convert("RGB")

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True)
