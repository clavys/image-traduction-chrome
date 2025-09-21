from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
import os
from typing import Optional
import sys
from PIL import Image, ImageDraw, ImageFont
import uvicorn
import time
import numpy as np

# Configuration des traducteurs
TRANSLATOR_CONFIG = {
    'default': 'google',  # Traducteur par défaut
    'deepl_api_key': "b3be89a4-31d3-4c97-a772-2481cf349dc8:fx",  
    'fallback_to_google': True  # Utiliser Google si DeepL échoue
}

# Classe pour gérer plusieurs traducteurs
class TranslatorManager:
    def __init__(self):
        self.translators = {}
        self.current_translator = None
        self.stats = {
            'google_requests': 0,
            'deepl_requests': 0,
            'failures': 0
        }
    
    def initialize_google(self):
        """Initialise Google Translate"""
        try:
            from modules.translators.trans_google import GoogleTranslator
            self.translators['google'] = GoogleTranslator()
            print("✅ Google Translate initialisé")
            return True
        except Exception as e:
            print(f"❌ Erreur Google Translate: {e}")
            return False
    
    def initialize_deepl(self, api_key: str):
        """Initialise DeepL avec clé API"""
        if not api_key:
            print("Pas de clé API DeepL fournie")
            return False
        
        try:
            from modules.translators.trans_deepl import DeeplTranslator
            # Initialiser avec les langues par défaut requises
            deepl_instance = DeeplTranslator(lang_source='日本語', lang_target='English')
            deepl_instance.params['api_key'] = api_key  # Configurer la clé via params
            self.translators['deepl'] = deepl_instance
            print("DeepL initialisé")
            return True
        except Exception as e:
            print(f"Erreur DeepL: {e}")
            return False
    
    def get_translator(self, preferred: str = None):
        """Récupère le traducteur préféré ou par défaut"""
        if preferred and preferred in self.translators:
            return self.translators[preferred], preferred
        
        # Ordre de préférence
        for service in ['deepl', 'google']:
            if service in self.translators:
                return self.translators[service], service
        
        return None, None
    
    def translate(self, text: str, target_language: str, preferred_service: str = None):
        """Traduit avec fallback automatique"""
        translator, service = self.get_translator(preferred_service)
        
        if not translator:
            raise Exception("Aucun traducteur disponible")
        
        try:
            if service == 'deepl':
                # Mapper les codes de langue vers les noms utilisés par BallonsTranslator
                lang_mapping = {
                    'ja': '日本語',
                    'en': 'English', 
                    'fr': 'Français',
                    'es': 'Español'
                }
                
                # Configurer les langues pour cette traduction
                translator.lang_source = lang_mapping.get('ja', '日本語')  # Source par défaut japonais
                translator.lang_target = lang_mapping.get(target_language, 'English')
                
                # Utiliser la méthode de traduction de BallonsTranslator
                result = translator._translate([text])
                translation = result[0] if result else text
            else:
                # Google Translate (méthode standard)
                translation = translator.translate(text, target_language=target_language)
            
            self.stats[f'{service}_requests'] += 1
            return translation
        except Exception as e:
            print(f"Erreur {service}: {e}")
            self.stats['failures'] += 1
            
            # Fallback vers Google si DeepL échoue
            if service == 'deepl' and 'google' in self.translators and TRANSLATOR_CONFIG['fallback_to_google']:
                try:
                    print("Fallback vers Google Translate...")
                    result = self.translators['google'].translate(text, target_language=target_language)
                    self.stats['google_requests'] += 1
                    return result
                except Exception as e2:
                    print(f"Erreur fallback Google: {e2}")
                    raise e2
            
            raise e
    
    def get_stats(self):
        """Statistiques d'utilisation"""
        return self.stats

# Instance globale du gestionnaire
translator_manager = TranslatorManager()

class ModuleCache:
    """Cache pour éviter de réinitialiser les modules à chaque requête"""
    def __init__(self):
        self._cached_modules = {}
        self._initialization_times = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_module(self, module_name):
        """Récupérer un module du cache"""
        if module_name in self._cached_modules:
            self.cache_hits += 1
            print(f"Cache HIT pour {module_name} (hits: {self.cache_hits})")
            return self._cached_modules[module_name]
        
        self.cache_misses += 1
        print(f"Cache MISS pour {module_name} (misses: {self.cache_misses})")
        return None
    
    def set_module(self, module_name, module_instance, init_time=0):
        """Mettre un module en cache"""
        self._cached_modules[module_name] = module_instance
        self._initialization_times[module_name] = init_time
        print(f"Module {module_name} mis en cache (init: {init_time:.2f}s)")
    
    def get_stats(self):
        """Statistiques du cache"""
        return {
            'cached_modules': list(self._cached_modules.keys()),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_ratio': self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        }
    
module_cache = ModuleCache()

# Fonction helper pour obtenir les modules avec cache
def get_cached_module(module_name):
    """Obtenir un module depuis le cache ou ballons_modules"""
    # Essayer le cache d'abord
    cached_module = module_cache.get_module(module_name)
    if cached_module:
        return cached_module
    
    # Fallback vers ballons_modules si pas en cache
    if module_name in ballons_modules:
        module_instance = ballons_modules[module_name]
        module_cache.set_module(module_name, module_instance)
        return module_instance
    
    print(f"Module {module_name} non disponible")
    return None

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

# Fonction d'initialisation à ajouter dans le lifespan
async def initialize_translators():
    """Initialise les traducteurs disponibles"""
    global translator_manager
    
    print("Initialisation des traducteurs...")
    
    # Google Translate (toujours disponible)
    translator_manager.initialize_google()
    
    # DeepL (si clé API disponible)
    deepl_key = os.getenv('DEEPL_API_KEY') or TRANSLATOR_CONFIG.get('deepl_api_key')
    
    # AJOUTEZ CES LIGNES DE DEBUG :
    print(f"Debug - Clé DeepL trouvée: {'Oui' if deepl_key else 'Non'}")
    if deepl_key:
        print(f"Debug - Clé (masquée): {deepl_key[:8]}...")
        print("Debug - Tentative d'initialisation DeepL...")
        result = translator_manager.initialize_deepl(deepl_key)
        print(f"Debug - Résultat initialisation: {result}")
    else:
        print("Clé DeepL non configurée, utilisation de Google uniquement")
    
    print(f"Traducteurs disponibles: {list(translator_manager.translators.keys())}")

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
                await initialize_translators()
                # Garder ballons_modules['translator'] pour compatibilité
                if 'google' in translator_manager.translators:
                    ballons_modules['translator'] = translator_manager.translators['google']
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


            #  NOUVELLE OPTIMISATION : Redimensionner si trop grande
            original_size = (image.width, image.height)
            max_size = 1024  # Taille maximum pour le traitement
            
            if max(image.width, image.height) > max_size:
                # Calculer le ratio pour garder les proportions
                ratio = min(max_size / image.width, max_size / image.height)
                new_width = int(image.width * ratio)
                new_height = int(image.height * ratio)
                
                print(f"📏 Redimensionnement: {original_size} -> ({new_width}, {new_height})")
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                print(f"📏 Taille OK: {original_size}")
                
            print(f"📸 Image traitée: {image.width}x{image.height}, mode: {image.mode}")
                
                
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

# Endpoint pour configurer DeepL à chaud
@app.post("/configure-deepl")
async def configure_deepl(api_key: str):
    """Configure DeepL avec une nouvelle clé API"""
    global translator_manager
    
    if translator_manager.initialize_deepl(api_key):
        TRANSLATOR_CONFIG['deepl_api_key'] = api_key
        return {"success": True, "message": "DeepL configuré avec succès"}
    else:
        return {"success": False, "message": "Erreur configuration DeepL"}

# Endpoint pour obtenir les stats
@app.get("/translator-stats")
async def get_translator_stats():
    """Statistiques des traducteurs"""
    return {
        "available_translators": list(translator_manager.translators.keys()),
        "default_translator": TRANSLATOR_CONFIG['default'],
        "stats": translator_manager.get_stats()
    }

# Fonctions utilitaires pour le workflow BallonsTranslator width height et area ajustables
def filter_small_text_blocks(blk_list, min_area=500):
    """Filtrer les blocs de texte trop petits pour économiser du temps"""
    if not blk_list:
        return blk_list
    
    filtered_blocks = []
    removed_count = 0
    
    for blk in blk_list:
        if hasattr(blk, 'xyxy'):
            x1, y1, x2, y2 = blk.xyxy
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Filtrer par aire ET par dimensions minimum
            if area >= min_area and width >= 20 and height >= 10:
                filtered_blocks.append(blk)
            else:
                removed_count += 1
                print(f"🚫 Zone ignorée: {width}x{height}px (aire: {area})")
        else:
            # Garder les blocs sans coordonnées (au cas où)
            filtered_blocks.append(blk)
    
    print(f"📊 Filtrage: {len(blk_list)} -> {len(filtered_blocks)} zones ({removed_count} supprimées)")
    return filtered_blocks


# Fonction pour la traduction par batch
def batch_translate_texts(translator, texts, target_lang, max_batch_size=5):
    """Traduire plusieurs textes en une seule requête"""
    if not texts or len(texts) == 0:
        return []
    
    if len(texts) == 1:
        # Un seul texte, traduction normale
        try:
            return [translator.translate(texts[0], target_language=target_lang)]
        except Exception as e:
            print(f"Erreur traduction unique: {e}")
            return [f"[ERREUR] {texts[0]}"]
    
    # Plusieurs textes, essayer le batch
    if len(texts) <= max_batch_size:
        try:
            # Combiner les textes avec des séparateurs uniques
            separator = "|||SEPARATOR|||"
            combined_text = separator.join(f"[{i}] {text}" for i, text in enumerate(texts))
            
            # Vérifier que ce n'est pas trop long
            if len(combined_text) > 4000:  # Limite de sécurité
                print("Texte combiné trop long, traduction individuelle")
                return translate_individually(translator, texts, target_lang)
            
            print(f"Traduction batch de {len(texts)} textes...")
            batch_result = translator.translate(combined_text, target_language=target_lang)
            
            # Parser le résultat
            parsed_results = parse_batch_result(batch_result, separator, len(texts))
            
            if len(parsed_results) == len(texts):
                print(f"Batch réussi: {len(parsed_results)} traductions")
                return parsed_results
            else:
                print("Parsing batch échoué, fallback individuel")
                return translate_individually(translator, texts, target_lang)
                
        except Exception as e:
            print(f"Erreur batch: {e}, fallback individuel")
            return translate_individually(translator, texts, target_lang)
    else:
        # Trop de textes, traduction individuelle
        return translate_individually(translator, texts, target_lang)

def parse_batch_result(batch_result, separator, expected_count):
    """Parser le résultat de traduction batch"""
    try:
        # Séparer par le séparateur
        parts = batch_result.split(separator)
        
        results = []
        for part in parts:
            # Enlever le préfixe [0], [1], etc.
            import re
            cleaned = re.sub(r'^\[\d+\]\s*', '', part.strip())
            if cleaned:
                results.append(cleaned)
        
        return results[:expected_count]  # Limiter au nombre attendu
        
    except Exception as e:
        print(f"Erreur parsing batch: {e}")
        return []

def translate_individually(translator, texts, target_lang):
    """Fallback: traduction individuelle"""
    results = []
    for text in texts:
        try:
            translation = translator.translate(text, target_language=target_lang)
            results.append(translation)
        except Exception as e:
            print(f"Erreur traduction '{text}': {e}")
            results.append(f"[ERREUR] {text}")
    return results

# Modifier la fonction translate_image_ballons_style
async def translate_image_ballons_style(image, request):
    """Workflow BallonsTranslator avec traduction par batch optimisée"""
    try:
        print("Workflow BallonsTranslator avec batch translation")
        
        img_array = np.array(image)
        im_h, im_w = img_array.shape[:2]
        
        # 1. Détection
        detector = get_cached_module('detector')
        if not detector:
            return add_debug_info(image, "Détecteur non disponible")
        
        blk_list = []
        print("Détection des zones de texte...")
        detection_start = time.time()
        mask, blk_list = detector.detect(img_array, blk_list)
        detection_time = time.time() - detection_start
        print(f"{len(blk_list)} TextBlocks détectés en {detection_time:.2f}s")
        
        if not blk_list:
            return add_debug_info(image, "Aucune zone de texte détectée")
        
        # 2. OCR
        ocr = get_cached_module('ocr')
        if ocr:
            ocr_start = time.time()
            try:
                if hasattr(ocr, '_ocr_blk_list'):
                    ocr._ocr_blk_list(img_array, blk_list)
                ocr_time = time.time() - ocr_start
                print(f"OCR terminé en {ocr_time:.2f}s")
            except Exception as e:
                print(f"Erreur OCR: {e}")
        
        # 3. TRADUCTION AVEC GESTIONNAIRE MULTIPLE
        translation_start = time.time()
        
        # Collecter tous les textes
        texts_to_translate = []
        text_indices = []
        
        for i, blk in enumerate(blk_list):
            text = blk.get_text()
            if text and text.strip():
                texts_to_translate.append(text.strip())
                text_indices.append(i)
        
        if texts_to_translate:
            # Déterminer le service préféré depuis la requête
            preferred_service = getattr(request, 'translator', 'google')
            if preferred_service not in translator_manager.translators:
                preferred_service = TRANSLATOR_CONFIG['default']
            
            print(f"Traduction de {len(texts_to_translate)} textes avec {preferred_service}...")
            
            # Traduction par batch avec le gestionnaire
            translated_texts = []
            for text in texts_to_translate:
                try:
                    translation = translator_manager.translate(
                        text, 
                        request.target_lang,
                        preferred_service
                    )
                    translated_texts.append(translation)
                except Exception as e:
                    print(f"Erreur traduction '{text}': {e}")
                    translated_texts.append(f"[ERREUR] {text}")
            
            # Assigner les traductions
            for idx, translated_text in enumerate(translated_texts):
                if idx < len(text_indices):
                    blk_idx = text_indices[idx]
                    original_text = texts_to_translate[idx]
                    blk_list[blk_idx].translation = translated_text
                    print(f"'{original_text}' -> '{translated_text}'")
            
            translation_time = time.time() - translation_start
            print(f"{len(translated_texts)} textes traduits en {translation_time:.2f}s")
        
        # 4. Rendu final
        inpainter = get_cached_module('inpainter')
        if inpainter and any(hasattr(blk, 'translation') and blk.translation for blk in blk_list):
            print("Rendu final...")
            render_start = time.time()
            result = render_ballons_result(image, img_array, blk_list, mask)
            render_time = time.time() - render_start
            print(f"Rendu terminé en {render_time:.2f}s")
            return result
        else:
            return render_ballons_overlay(image, blk_list)
        
    except Exception as e:
        print(f"Erreur workflow: {e}")
        import traceback
        traceback.print_exc()
        return add_debug_info(image, f"Erreur: {str(e)}")

def get_font(size=18, bubble_width=None, bubble_height=None, text_length=None):
    """Police avec taille adaptative simple"""
    
    if bubble_width and bubble_height:
        # Taille basée sur la plus petite dimension de la bulle
        min_dimension = min(bubble_width, bubble_height)
        
        if min_dimension > 150:
            font_size = 20
        elif min_dimension > 100:
            font_size = 16
        elif min_dimension > 60:
            font_size = 14
        else:
            font_size = 12
        
        # Réduire légèrement si le texte est très long
        if text_length and text_length > 30:
            font_size = max(10, font_size - 2)
    else:
        font_size = size
    
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
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

        for blk in blk_list:
            if blk.translation and hasattr(blk, 'xyxy'):
                x1, y1, x2, y2 = map(int, blk.xyxy)
                max_width = x2 - x1
                max_height = y2 - y1
                
                # Font adaptative pour chaque bulle
                font = get_font(18, max_width, max_height, len(blk.translation))
                ascent, descent = font.getmetrics()
                line_spacing = ascent + descent + 2
                
                lines = wrap_text(blk.translation, font, max_width, draw)
                total_height = len(lines) * line_spacing
                y_text = y1 + (max_height - total_height) // 2

                for line in lines:
                    w = draw.textlength(line, font=font)
                    x_text = x1 + (max_width - w) // 2
                    draw_text_with_outline(draw, (x_text, y_text), line, font)
                    y_text += line_spacing
                    
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
