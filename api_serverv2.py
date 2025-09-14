import io
import base64
import threading
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from ui.canvas import Canvas

# ⚙️ Lock pour Qt
qt_lock = threading.Lock()

# 🔥 FastAPI
app = FastAPI(title="BallonsTranslator API")

# 📦 Pydantic pour la requête
class TranslateRequest(BaseModel):
    image_base64: str
    source_lang: str = "ja"
    target_lang: str = "en"
    translator: str = "google"

# 🖌 Fonction principale de traduction via Canvas
def translate_image_ballons_style(img_array: np.ndarray) -> str:
    print(f"📸 Image reçue: {img_array.shape[1]}x{img_array.shape[0]}")
    try:
        canvas = Canvas()
        with qt_lock:
            # Rendu de l'image traduite
            pil_image = canvas.render_result_img(img_array)
        
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        print("✅ Image traduite avec Canvas")
        return img_base64
    except Exception as e:
        print(f"❌ Erreur dans translate_image_ballons_style: {e}")
        raise

# 🔧 Endpoint de test API
@app.get("/health")
async def health():
    return {"status": "healthy"}

# 🌐 Endpoint principal de traduction
@app.post("/translate")
async def translate(data: TranslateRequest):
    try:
        # Décoder l'image
        img_bytes = base64.b64decode(data.image_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(img)

        print("🎯 Workflow BallonsTranslator natif...")

        # Traduction avec Canvas
        translated_base64 = translate_image_ballons_style(img_array)

        print("✅ Traduction terminée")
        return {
            "success": True,
            "translated_image_base64": translated_base64,
            "processing_time": 0  # tu peux mesurer si besoin
        }
    except Exception as e:
        print(f"❌ Erreur workflow: {e}")
        return {"success": False, "error": str(e)}

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
