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

# ==================================================================================
# WORKFLOW PRINCIPAL AVEC RENDU NATIF BALLONSTRANSLATOR
# ==================================================================================

async def translate_image_ballons_style(image, request):
    """Workflow BallonsTranslator avec rendu natif PERFECTIONNÉ"""
    try:
        print("🔄 Workflow BallonsTranslator natif avec RENDU PARFAIT")
        
        # Conversion PIL -> numpy
        img_array = np.array(image)
        im_h, im_w = img_array.shape[:2]
        
        # 1. Détection des zones de texte
        detector = ballons_modules['detector']
        blk_list = []
        
        print("🔍 Détection des zones de texte...")
        mask, blk_list = detector.detect(img_array, blk_list)
        print(f"📍 {len(blk_list)} TextBlocks détectés")
        
        if not blk_list:
            return add_debug_info(image, "Aucune zone de texte détectée")
        
        # 2. OCR avec la méthode native BallonsTranslator
        if 'ocr' in ballons_modules:
            ocr = ballons_modules['ocr']
            print("📖 OCR avec méthode interne BallonsTranslator...")
            
            try:
                # Utiliser la méthode OCR exacte de BallonsTranslator
                if hasattr(ocr, '_ocr_blk_list'):
                    print("    Utilisation de _ocr_blk_list (méthode interne)")
                    ocr._ocr_blk_list(img_array, blk_list)
                elif hasattr(ocr, 'run_ocr'):
                    print("    Utilisation de run_ocr")
                    result = ocr.run_ocr(img_array, blk_list)
                    if isinstance(result, list):
                        blk_list = result
                else:
                    # Fallback manuel si nécessaire
                    print("    Méthode OCR manuelle")
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
        
        # 3. Traduction
        if 'translator' in ballons_modules:
            translator = ballons_modules['translator']
            print("🌐 Traduction des TextBlocks...")
            
            translated_count = 0
            for blk in blk_list:
                text = blk.get_text()
                if text and text.strip():
                    try:
                        translation = translator.translate(text, target_language=request.target_lang)
                        blk.translation = translation
                        translated_count += 1
                        print(f"🔄 '{text}' -> '{translation}'")
                    except Exception as e:
                        print(f"⚠️ Erreur traduction: {e}")
                        blk.translation = f"[ERREUR] {text}"
            
            print(f"📝 {translated_count} blocs traduits")
        
        # 4. *** RENDU NATIF BALLONSTRANSLATOR - LA SOLUTION PARFAITE ***
        print("🎨 RENDU NATIF BallonsTranslator - Qualité PARFAITE...")
        return await render_with_ballons_perfect_method(img_array, blk_list, mask, request)
        
    except Exception as e:
        print(f"❌ Erreur workflow BallonsTranslator: {e}")
        import traceback
        traceback.print_exc()
        return add_debug_info(image, f"Erreur workflow: {str(e)}")

# ==================================================================================
# SYSTÈME DE RENDU NATIF BALLONSTRANSLATOR (5 MÉTHODES)
# ==================================================================================

async def render_with_ballons_perfect_method(img_array, blk_list, mask, request):
    """LA MÉTHODE PARFAITE - Utilise l'export natif exact de BallonsTranslator"""
    
    # Essayer les méthodes dans l'ordre de priorité
    methods = [
        ("ModuleManager + Canvas Export", render_with_module_manager),
        ("Classe Principale BallonsTranslator", render_with_main_class),
        ("Pipeline d'Export Natif", render_with_export_pipeline),
        ("TextRender + Inpainter", render_with_textrender),
        ("Sauvegarde Native", render_with_save_method)
    ]
    
    for method_name, method_func in methods:
        try:
            print(f"🎯 Tentative: {method_name}")
            result = await method_func(img_array, blk_list, mask, request)
            if result is not None:
                print(f"✅ Succès avec: {method_name}")
                return result
        except Exception as e:
            print(f"❌ {method_name} échoué: {e}")
            continue
    
    # Si toutes les méthodes échouent, utiliser le fallback optimisé
    print("📋 Toutes les méthodes natives ont échoué, utilisation du fallback optimisé")
    return render_fallback_optimized(Image.fromarray(img_array), blk_list)

# MÉTHODES DE RENDU NATIF (dans l'ordre de priorité)

async def render_with_module_manager(img_array, blk_list, mask, request):
    """Méthode #1 - ModuleManager + Canvas (la plus proche de l'application)"""
    try:
        from ui.module_manager import ModuleManager
        from ui.canvas import Canvas
        from utils.imgproc_utils import qimage2ndarray, ndarray2qimage
        
        # Créer le gestionnaire de modules comme dans l'app
        module_manager = ModuleManager()
        module_manager.detector = ballons_modules.get('detector')
        module_manager.ocr = ballons_modules.get('ocr')
        module_manager.translator = ballons_modules.get('translator')
        module_manager.inpainter = ballons_modules.get('inpainter')
        
        # Créer un Canvas et charger l'image
        canvas = Canvas()
        qimage = ndarray2qimage(img_array)
        canvas.load_image(qimage)
        canvas.textblk_lst = blk_list
        
        # Export avec le système natif
        result_qimage = canvas.get_export_image(
            inpaint=True,
            draw_text=True,
            textblock_mask=True
        )
        
        result_array = qimage2ndarray(result_qimage)
        return Image.fromarray(result_array)
    
    except ImportError:
        raise Exception("ModuleManager non disponible")

async def render_with_main_class(img_array, blk_list, mask, request):
    """Méthode #2 - Classe principale BallonsTranslator"""
    try:
        from ballontranslator import BallonsTranslator
        
        bt = BallonsTranslator()
        bt.detector = ballons_modules.get('detector')
        bt.ocr = ballons_modules.get('ocr')
        bt.translator = ballons_modules.get('translator')
        bt.inpainter = ballons_modules.get('inpainter')
        
        result_img_array = bt.translate_img(
            img_array,
            blk_list=blk_list,
            target_lang=request.target_lang,
            inpaint=True,
            font_size_offset=0,
            font_color=(0, 0, 0),
            stroke_width=0,
            stroke_color=(255, 255, 255),
            auto_font_size=True,
            auto_text_direction=True,
            preserve_formatting=True
        )
        
        return Image.fromarray(result_img_array.astype(np.uint8))
    
    except ImportError:
        raise Exception("Classe principale BallonsTranslator non disponible")

async def render_with_export_pipeline(img_array, blk_list, mask, request):
    """Méthode #3 - Pipeline d'export natif"""
    try:
        from utils.imgproc_utils import ImgTranslationPipeline
        
        pipeline = ImgTranslationPipeline()
        pipeline.detector = ballons_modules.get('detector')
        pipeline.ocr = ballons_modules.get('ocr')
        pipeline.translator = ballons_modules.get('translator')
        pipeline.inpainter = ballons_modules.get('inpainter')
        
        result_img_array = pipeline.run_translation_pipeline(
            img_array,
            blk_list=blk_list,
            target_lang=request.target_lang,
            skip_detection=True,
            skip_ocr=True,
            skip_translation=True,
            render_text=True,
            inpaint=True
        )
        
        return Image.fromarray(result_img_array.astype(np.uint8))
    
    except ImportError:
        raise Exception("ImgTranslationPipeline non disponible")

async def render_with_textrender(img_array, blk_list, mask, request):
    """Méthode #4 - TextRender + Inpainter natifs"""
    try:
        from utils.textrender import TextRender
        from utils.fontformat import FontFormat
        
        # Inpainting d'abord
        if 'inpainter' in ballons_modules:
            inpainter = ballons_modules['inpainter']
            mask_array = np.zeros(img_array.shape[:2], dtype=np.uint8)
            
            for blk in blk_list:
                if hasattr(blk, 'xyxy') and blk.translation:
                    x1, y1, x2, y2 = map(int, blk.xyxy)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_array.shape[1], x2), min(img_array.shape[0], y2)
                    mask_array[y1:y2, x1:x2] = 255
            
            inpainted_array = inpainter.inpaint(img_array, mask_array)
        else:
            inpainted_array = img_array.copy()
        
        # Rendu du texte avec TextRender natif
        text_render = TextRender()
        result_array = inpainted_array.copy()
        
        for blk in blk_list:
            if not blk.translation or not hasattr(blk, 'xyxy'):
                continue
            
            font_format = getattr(blk, 'fontformat', FontFormat())
            rendered_blk = text_render.render_textblock(
                blk, 
                font_format,
                result_array.shape[1],
                result_array.shape[0]
            )
            
            if rendered_blk is not None:
                text_render.paste_textblock_on_image(result_array, rendered_blk, blk)
        
        return Image.fromarray(result_array.astype(np.uint8))
    
    except ImportError:
        raise Exception("TextRender non disponible")

async def render_with_save_method(img_array, blk_list, mask, request):
    """Méthode #5 - Fonction de sauvegarde native"""
    try:
        import tempfile
        import os
        from ui.io_thread import save_page_as_image
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            save_page_as_image(
                img_array,
                blk_list,
                temp_path,
                inpainter=ballons_modules.get('inpainter'),
                render_text=True,
                inpaint=True,
                font_size_offset=0,
                background_color='white',
                text_color='black',
                quality=95
            )
            
            result_image = Image.open(temp_path)
            return result_image
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except ImportError:
        raise Exception("Fonction de sauvegarde non disponible")

# ==================================================================================
# RENDU FALLBACK OPTIMISÉ
# ==================================================================================

def render_fallback_optimized(image, blk_list):
    """Rendu manuel optimisé basé sur les spécifications BallonsTranslator"""
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    
    print("🎨 Utilisation du rendu fallback optimisé")
    
    for blk in blk_list:
        if not blk.translation or not hasattr(blk, 'xyxy'):
            continue
            
        x1, y1, x2, y2 = map(int, blk.xyxy)
        zone_width = x2 - x1
        zone_height = y2 - y1
        
        # Utiliser les propriétés détectées par BallonsTranslator
        font_size = getattr(blk, 'font_size', 16)
        if hasattr(blk, 'fontformat') and hasattr(blk.fontformat, 'font_size'):
            font_size = int(blk.fontformat.font_size)
        
        # Adapter la taille à la zone (méthode BallonsTranslator)
        font_size = min(font_size, zone_height // 3, zone_width // 6)
        font_size = max(font_size, 10)
        
        # Propriétés d'orientation BallonsTranslator
        vertical = getattr(blk, 'vertical', False)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Préparation du texte selon l'orientation
        translation = blk.translation
        if vertical:
            # Texte vertical (style manga japonais)
            lines = list(translation)  # Un caractère par ligne
        else:
            # Texte horizontal avec découpage intelligent
            words = translation.split()
            lines = []
            current_line = ""
            
            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                text_width = draw.textlength(test_line, font=font)
                
                if text_width <= zone_width - 20:  # Marge de 20px
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                        current_line = word
                    else:
                        lines.append(word)
            
            if current_line:
                lines.append(current_line)
        
        # Limite de lignes selon la zone
        max_lines = max(1, zone_height // (font_size + 6))
        lines = lines[:max_lines]
        
        # Positionnement centré (méthode BallonsTranslator)
        total_text_height = len(lines) * (font_size + 4)
        start_y = y1 + (zone_height - total_text_height) // 2
        
        # Fond blanc pour lisibilité
        bg_margin = 4
        bg_x1 = x1 + bg_margin
        bg_y1 = start_y - bg_margin
        bg_x2 = x2 - bg_margin
        bg_y2 = start_y + total_text_height + bg_margin
        
        # Dessiner fond blanc semi-transparent
        overlay = Image.new('RGBA', result_image.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(255, 255, 255, 220))
        
        result_image = Image.alpha_composite(
            result_image.convert('RGBA'), 
            overlay
        ).convert('RGB')
        
        draw = ImageDraw.Draw(result_image)
        
        # Dessiner le texte ligne par ligne
        for i, line in enumerate(lines):
            if vertical:
                # Texte vertical centré horizontalement
                text_x = x1 + (zone_width - font_size) // 2
                text_y = start_y + i * (font_size + 2)
            else:
                # Texte horizontal centré
                text_width = draw.textlength(line, font=font)
                text_x = x1 + (zone_width - text_width) // 2
                text_y = start_y + i * (font_size + 4)
            
            # Contraintes des limites
            text_x = max(5, min(text_x, result_image.width - 50))
            text_y = max(5, min(text_y, result_image.height - font_size - 5))
            
            # Dessiner le texte en noir
            draw.text((text_x, text_y), line, fill="black", font=font)
    
    # Signature
    small_font = get_font(size=8)
    draw.text((5, result_image.height - 15), "BallonsTranslator Integration (Fallback)", fill=(120, 120, 120), font=small_font)
    
    return result_image

# ==================================================================================
# FONCTIONS UTILITAIRES
# ==================================================================================

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

# ==================================================================================
# DÉMARRAGE DU SERVEUR
# ==================================================================================

if __name__ == "__main__":
    print("🚀 Démarrage Manga Translator API avec BallonsTranslator Rendu Natif...")
    print("📚 Documentation: http://localhost:8000/docs")
    print("💚 Health check: http://localhost:8000/health")
    print("🎯 Interface: http://localhost:8000/")
    print("🔌 Extension Chrome compatible")
    print("🎨 Rendu natif BallonsTranslator intégré (5 méthodes)")
    
    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
