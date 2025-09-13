// content.js - Script injecté dans chaque page web
console.log('🚀 Manga Translator extension loaded');

// Configuration
const API_BASE_URL = 'http://localhost:8000';
const TRANSLATION_KEY = 'manga-translator-active';

// État de l'extension
let isTranslationActive = false;
let processedImages = new WeakSet();

// Styles CSS pour les overlays
const CSS_STYLES = `
  .manga-translator-overlay {
    position: absolute;
    background: rgba(255, 0, 0, 0.8);
    color: white;
    padding: 2px 6px;
    font-size: 12px;
    font-weight: bold;
    border-radius: 3px;
    z-index: 10000;
    pointer-events: none;
    font-family: Arial, sans-serif;
  }
  
  .manga-translator-processing {
    background: rgba(255, 165, 0, 0.8);
  }
  
  .manga-translator-translated {
    background: rgba(0, 128, 0, 0.8);
  }
  
  .manga-translator-button {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 10001;
    background: #4285f4;
    color: white;
    border: none;
    padding: 10px 15px;
    border-radius: 5px;
    cursor: pointer;
    font-weight: bold;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
  }
  
  .manga-translator-button:hover {
    background: #3367d6;
  }
  
  .manga-translator-button.active {
    background: #34a853;
  }
`;

// Ajouter les styles CSS
function injectStyles() {
  if (document.getElementById('manga-translator-styles')) return;
  
  const styleSheet = document.createElement('style');
  styleSheet.id = 'manga-translator-styles';
  styleSheet.textContent = CSS_STYLES;
  document.head.appendChild(styleSheet);
}

// Créer le bouton de contrôle
function createControlButton() {
  if (document.getElementById('manga-translator-btn')) return;
  
  const button = document.createElement('button');
  button.id = 'manga-translator-btn';
  button.className = 'manga-translator-button';
  button.textContent = 'Manga Translator: OFF';
  
  button.addEventListener('click', toggleTranslation);
  document.body.appendChild(button);
  
  // Charger l'état sauvegardé
  chrome.storage.local.get([TRANSLATION_KEY], (result) => {
    if (result[TRANSLATION_KEY]) {
      activateTranslation();
    }
  });
}

// Activer/désactiver la traduction
function toggleTranslation() {
  if (isTranslationActive) {
    deactivateTranslation();
  } else {
    activateTranslation();
  }
}

function activateTranslation() {
  isTranslationActive = true;
  const button = document.getElementById('manga-translator-btn');
  if (button) {
    button.textContent = 'Manga Translator: ON';
    button.classList.add('active');
  }
  
  // Sauvegarder l'état
  chrome.storage.local.set({[TRANSLATION_KEY]: true});
  
  // Scanner toutes les images
  scanForImages();
}

function deactivateTranslation() {
  isTranslationActive = false;
  const button = document.getElementById('manga-translator-btn');
  if (button) {
    button.textContent = 'Manga Translator: OFF';
    button.classList.remove('active');
  }
  
  // Sauvegarder l'état
  chrome.storage.local.set({[TRANSLATION_KEY]: false});
  
  // Supprimer tous les overlays
  document.querySelectorAll('.manga-translator-overlay').forEach(el => el.remove());
}

// Scanner les images sur la page
function scanForImages() {
  if (!isTranslationActive) return;
  
  const images = document.querySelectorAll('img');
  console.log(`📸 Found ${images.length} images on page`);
  
  images.forEach((img, index) => {
    if (!processedImages.has(img) && shouldTranslateImage(img)) {
      processedImages.add(img);
      setTimeout(() => processImage(img), index * 1000); // Étaler les requêtes
    }
  });
}

// Vérifier si une image doit être traduite
function shouldTranslateImage(img) {
  // Filtres pour éviter les petites images, logos, etc.
  const minWidth = 100;
  const minHeight = 100;
  
  if (img.naturalWidth < minWidth || img.naturalHeight < minHeight) {
    return false;
  }
  
  // Éviter les images système
  const skipPatterns = ['logo', 'icon', 'avatar', 'button'];
  const src = img.src.toLowerCase();
  
  if (skipPatterns.some(pattern => src.includes(pattern))) {
    return false;
  }
  
  return true;
}

// Traiter une image
async function processImage(img) {
  try {
    // Ajouter overlay de traitement
    const overlay = createOverlay(img, 'Processing...', 'manga-translator-processing');
    
    // Convertir l'image en base64
    const base64Image = await imageToBase64(img);
    
    // Envoyer à l'API
    const response = await fetch(`${API_BASE_URL}/translate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_base64: base64Image,
        source_lang: 'ja',
        target_lang: 'en',
        translator: 'google'
      })
    });
    
    const result = await response.json();
    
    if (result.success) {
      // Créer une nouvelle image au lieu de remplacer directement
      const newImg = new Image();
      newImg.onload = () => {
        // Remplacer l'ancienne image
        img.src = newImg.src;
        updateOverlay(overlay, 'Translated!', 'manga-translator-translated');
        
        // Supprimer l'overlay après 2 secondes
        setTimeout(() => overlay.remove(), 2000);
        
        console.log(`✅ Image translated in ${result.processing_time.toFixed(2)}s`);
      };
      
      newImg.onerror = () => {
        console.error('Failed to load translated image');
        updateOverlay(overlay, 'Load error', 'manga-translator-overlay');
        setTimeout(() => overlay.remove(), 3000);
      };
      
      // Charger l'image traduite
      newImg.src = `data:image/png;base64,${result.translated_image_base64}`;
      
    } else {
      updateOverlay(overlay, 'Error: ' + result.error, 'manga-translator-overlay');
      setTimeout(() => overlay.remove(), 3000);
    }
    
  } catch (error) {
    console.error('❌ Translation failed:', error);
    const overlay = document.querySelector(`[data-img-id="${img.dataset.imgId}"]`);
    if (overlay) {
      updateOverlay(overlay, 'Failed', 'manga-translator-overlay');
      setTimeout(() => overlay.remove(), 3000);
    }
  }
}

// Convertir image en base64
function imageToBase64(img) {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    
    img.crossOrigin = 'anonymous';
    
    img.onload = () => {
      ctx.drawImage(img, 0, 0);
      try {
        const base64 = canvas.toDataURL('image/png').split(',')[1];
        resolve(base64);
      } catch (error) {
        reject(error);
      }
    };
    
    img.onerror = reject;
    
    // Si l'image est déjà chargée
    if (img.complete) {
      ctx.drawImage(img, 0, 0);
      try {
        const base64 = canvas.toDataURL('image/png').split(',')[1];
        resolve(base64);
      } catch (error) {
        reject(error);
      }
    }
  });
}

// Créer un overlay sur l'image
function createOverlay(img, text, className = 'manga-translator-overlay') {
  const overlay = document.createElement('div');
  overlay.className = className;
  overlay.textContent = text;
  
  // ID unique pour retrouver l'overlay
  const imgId = Date.now() + Math.random();
  img.dataset.imgId = imgId;
  overlay.dataset.imgId = imgId;
  
  positionOverlay(overlay, img);
  document.body.appendChild(overlay);
  
  return overlay;
}

// Mettre à jour un overlay
function updateOverlay(overlay, text, className) {
  overlay.textContent = text;
  overlay.className = className;
}

// Positionner l'overlay sur l'image
function positionOverlay(overlay, img) {
  const rect = img.getBoundingClientRect();
  overlay.style.left = (rect.left + window.scrollX + 5) + 'px';
  overlay.style.top = (rect.top + window.scrollY + 5) + 'px';
}

// Observer les nouvelles images (lazy loading)
function observeNewImages() {
  const observer = new MutationObserver((mutations) => {
    if (!isTranslationActive) return;
    
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeName === 'IMG') {
          if (shouldTranslateImage(node)) {
            processedImages.add(node);
            setTimeout(() => processImage(node), 500);
          }
        } else if (node.querySelectorAll) {
          const images = node.querySelectorAll('img');
          images.forEach((img) => {
            if (!processedImages.has(img) && shouldTranslateImage(img)) {
              processedImages.add(img);
              setTimeout(() => processImage(img), 500);
            }
          });
        }
      });
    });
  });
  
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
}

// Initialisation
function init() {
  injectStyles();
  createControlButton();
  observeNewImages();
  
  // Scanner les images après un délai pour laisser la page se charger
  setTimeout(scanForImages, 2000);
}

// Démarrer quand le DOM est prêt
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

// Message depuis le popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'toggleTranslation') {
    toggleTranslation();
    sendResponse({active: isTranslationActive});
  }
});
