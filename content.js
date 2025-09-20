// content.js - Script injecté dans chaque page web
console.log('🚀 Manga Translator extension loaded');

// Configuration
const API_BASE_URL = 'http://localhost:8000';
const TRANSLATION_KEY = 'manga-translator-active';

// État de l'extension
let isTranslationActive = false;
let processedImages = new WeakSet();
let autoTranslateEnabled = true;

// Observer les changements de page (SPA, navigation AJAX)
let currentUrl = window.location.href;

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
  
  .manga-translator-button.auto {
    background: #9c27b0;
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
  button.textContent = 'Manga Translator: AUTO';
  
  button.addEventListener('click', toggleMode);
  document.body.appendChild(button);
  
  // Charger l'état et démarrer automatiquement
  chrome.storage.local.get([TRANSLATION_KEY, 'auto-translate'], (result) => {
    autoTranslateEnabled = result['auto-translate'] !== false; // Par défaut: true
    
    if (autoTranslateEnabled) {
      activateTranslation();
      updateButtonDisplay();
    }
  });
}

// Basculer entre les modes
function toggleMode() {
  if (!isTranslationActive && !autoTranslateEnabled) {
    // OFF -> AUTO
    autoTranslateEnabled = true;
    activateTranslation();
  } else if (isTranslationActive && autoTranslateEnabled) {
    // AUTO -> MANUAL
    autoTranslateEnabled = false;
    // Rester activé mais en mode manuel
  } else {
    // MANUAL -> OFF
    autoTranslateEnabled = false;
    deactivateTranslation();
  }
  
  updateButtonDisplay();
  
  // Sauvegarder les préférences
  chrome.storage.local.set({
    [TRANSLATION_KEY]: isTranslationActive,
    'auto-translate': autoTranslateEnabled
  });
}

// Mettre à jour l'affichage du bouton
function updateButtonDisplay() {
  const button = document.getElementById('manga-translator-btn');
  if (!button) return;
  
  if (isTranslationActive && autoTranslateEnabled) {
    button.textContent = 'Manga Translator: AUTO';
    button.className = 'manga-translator-button auto active';
  } else if (isTranslationActive && !autoTranslateEnabled) {
    button.textContent = 'Manga Translator: MANUAL';
    button.className = 'manga-translator-button active';
  } else {
    button.textContent = 'Manga Translator: OFF';
    button.className = 'manga-translator-button';
  }
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
  updateButtonDisplay();
  
  // Sauvegarder l'état
  chrome.storage.local.set({[TRANSLATION_KEY]: true});
  
  // Scanner toutes les images
  scanForImagesOptimized();
}

function deactivateTranslation() {
  isTranslationActive = false;
  updateButtonDisplay();
  
  // Sauvegarder l'état
  chrome.storage.local.set({[TRANSLATION_KEY]: false});
  
  // Supprimer tous les overlays
  document.querySelectorAll('.manga-translator-overlay').forEach(el => el.remove());
}

// Détecter les changements de page
function detectPageChange() {
  // Observer les changements d'URL (pour les SPA comme React)
  const observer = new MutationObserver(() => {
    if (currentUrl !== window.location.href) {
      currentUrl = window.location.href;
      console.log('🔄 Page change detected:', currentUrl);
      onPageChange();
    }
  });
  
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
  
  // Écouter les événements de navigation
  window.addEventListener('popstate', onPageChange);
  window.addEventListener('pushstate', onPageChange);
  window.addEventListener('replacestate', onPageChange);
  
  // Override pushState et replaceState pour les détecter
  const originalPushState = history.pushState;
  const originalReplaceState = history.replaceState;
  
  history.pushState = function(...args) {
    originalPushState.apply(history, args);
    setTimeout(onPageChange, 100);
  };
  
  history.replaceState = function(...args) {
    originalReplaceState.apply(history, args);
    setTimeout(onPageChange, 100);
  };
}

// Actions à effectuer lors d'un changement de page
function onPageChange() {
  console.log('📄 Page changed, checking translation settings...');
  
  // Réinitialiser les images traitées
  processedImages = new WeakSet();
  
  // Supprimer les anciens overlays
  document.querySelectorAll('.manga-translator-overlay').forEach(el => el.remove());
  
  // Si auto-traduction activée, démarrer automatiquement
  if (autoTranslateEnabled) {
    console.log('🚀 Auto-translating new page...');
    
    // Attendre que les images se chargent
    setTimeout(() => {
      if (!isTranslationActive) {
        activateTranslation();
      } else {
        scanForImagesOptimized();
      }
    }, 1500);
  }
}

// Observer les nouvelles images avec auto-traduction
function observeNewImages() {
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeName === 'IMG') {
          handleNewImage(node);
        } else if (node.querySelectorAll) {
          const images = node.querySelectorAll('img');
          images.forEach(handleNewImage);
        }
      });
    });
  });
  
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
}

// Gérer une nouvelle image détectée
function handleNewImage(img) {
  if (processedImages.has(img) || !shouldTranslateImage(img)) {
    return;
  }
  
  processedImages.add(img);
  
  // Si auto-traduction ou mode manuel activé
  if (autoTranslateEnabled || isTranslationActive) {
    if (!isTranslationActive) {
      activateTranslation();
    }
    setTimeout(() => processImageOptimized(img), 500);
  }
}

// Vérifier si une image doit être traduite
function shouldTranslateImage(img) {
  const minWidth = 100;
  const minHeight = 100;
  
  if (img.naturalWidth < minWidth || img.naturalHeight < minHeight) {
    return false;
  }
  
  const skipPatterns = ['logo', 'icon', 'avatar', 'button'];
  const src = img.src.toLowerCase();
  
  if (skipPatterns.some(pattern => src.includes(pattern))) {
    return false;
  }
  
  return true;
}

// Initialisation
function init() {
  injectStyles();
  createControlButton();
  observeNewImages();
  detectPageChange();
  
  // Scanner les images après un délai pour laisser la page se charger
  setTimeout(() => {
    // Si auto-traduction activée, démarrer automatiquement
    chrome.storage.local.get(['auto-translate'], (result) => {
      if (result['auto-translate'] !== false) {
        console.log('🚀 Auto-starting translation on page load...');
        if (!isTranslationActive) {
          activateTranslation();
        }
      }
    });
  }, 2000);
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

// Créer un overlay sur l'image
function createOverlay(img, text, className = 'manga-translator-overlay') {
  const overlay = document.createElement('div');
  overlay.className = className;
  overlay.textContent = text;
  
  const imgId = Date.now() + Math.random();
  img.dataset.imgId = imgId;
  overlay.dataset.imgId = imgId;
  
  const rect = img.getBoundingClientRect();
  overlay.style.position = 'absolute';
  overlay.style.left = (rect.left + window.scrollX + 5) + 'px';
  overlay.style.top = (rect.top + window.scrollY + 5) + 'px';
  
  document.body.appendChild(overlay);
  return overlay;
}

// Mettre à jour un overlay
function updateOverlay(overlay, text, className) {
  overlay.textContent = text;
  overlay.className = className;
}

// Convertir image en base64
async function imageToBase64(img) {
  try {
    const response = await chrome.runtime.sendMessage({
      action: 'fetchImage',
      url: img.src
    });
    
    if (response && response.success) {
      return response.base64;
    } else {
      throw new Error(response ? response.error : 'No response from background');
    }
  } catch (error) {
    console.warn('Background fetch failed, using test image');
    return 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==';
  }
}

// Filtrage intelligent des images
function shouldTranslateImageOptimized(img) {
    // Critères plus stricts pour éviter les images inutiles
    const minWidth = 120;
    const minHeight = 60;
    
    if (img.naturalWidth < minWidth || img.naturalHeight < minHeight) {
        return false;
    }
    
    // Éviter les images trop grandes (probablement des backgrounds)
    const maxArea = 2000 * 2000;
    if (img.naturalWidth * img.naturalHeight > maxArea) {
        console.log('🚫 Image trop grande ignorée:', img.naturalWidth, 'x', img.naturalHeight);
        return false;
    }
    
    // Patterns à éviter plus complets
    const skipPatterns = [
        'logo', 'icon', 'avatar', 'button', 'emoji', 'arrow',
        'background', 'bg-', 'header', 'footer', 'nav', 'menu',
        'thumb', 'preview', 'cover'
    ];
    
    const src = img.src.toLowerCase();
    const className = (img.className || '').toLowerCase();
    const alt = (img.alt || '').toLowerCase();
    const id = (img.id || '').toLowerCase();
    
    if (skipPatterns.some(pattern => 
        src.includes(pattern) || className.includes(pattern) || 
        alt.includes(pattern) || id.includes(pattern)
    )) {
        return false;
    }
    
    // Éviter les ratios extrêmes (banners, barres)
    const ratio = Math.max(img.naturalWidth, img.naturalHeight) / 
                  Math.min(img.naturalWidth, img.naturalHeight);
    if (ratio > 5) {
        console.log('🚫 Ratio extrême ignoré:', ratio.toFixed(1));
        return false;
    }
    
    // Vérifier si l'image est visible (éviter les images cachées)
    const rect = img.getBoundingClientRect();
    if (rect.width < 50 || rect.height < 30) {
        return false;
    }
    
    return true;
}

// Traitement avec timeout et gestion d'erreurs optimisée
async function processImageOptimized(img) {
    const maxRetries = 1;
    const timeout = 12000;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            console.log(`🔄 Traitement (${attempt}/${maxRetries}):`, img.src.substring(0, 40) + '...');
            
            const overlay = createOverlay(img, `Processing...`, 'manga-translator-processing');
            
            // Début du traitement avec timeout
            const startTime = Date.now();
            const translationPromise = performTranslation(img);
            const timeoutPromise = new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Timeout')), timeout)
            );
            
            const result = await Promise.race([translationPromise, timeoutPromise]);
            const processingTime = (Date.now() - startTime) / 1000;
            
            if (result.success) {
                await displayTranslatedImage(img, result, overlay);
                
                // Stats
                chrome.runtime.sendMessage({
                    action: 'updateStats',
                    data: { processingTime: processingTime }
                });
                
                console.log(`✅ Succès en ${processingTime.toFixed(2)}s`);
                return;
            } else {
                throw new Error(result.error);
            }
            
        } catch (error) {
            console.warn(`❌ Tentative ${attempt} échouée:`, error.message);
            
            if (attempt === maxRetries) {
                const overlay = createOverlay(img, 
                    `Failed: ${error.message.substring(0, 20)}...`, 
                    'manga-translator-overlay'
                );
                setTimeout(() => overlay.remove(), 3000);
            }
        }
    }
}

// Fonction de traduction avec settings en cache
let cachedSettings = null;

async function performTranslation(img) {
    // Settings en cache pour éviter les appels répétés au storage
    if (!cachedSettings) {
        cachedSettings = await new Promise(resolve => {
            chrome.storage.local.get([
                'api-url', 'source-lang', 'target-lang'
            ], resolve);
        });
    }
    
    const base64Image = await imageToBase64(img);
    const apiUrl = cachedSettings['api-url'] || 'http://localhost:8000';
    
    const response = await fetch(`${apiUrl}/translate`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            image_base64: base64Image,
            source_lang: cachedSettings['source-lang'] || 'ja',
            target_lang: cachedSettings['target-lang'] || 'en',
            translator: 'google'
        })
    });
    
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }
    
    return await response.json();
}

// Affichage plus fluide des résultats
async function displayTranslatedImage(originalImg, result, overlay) {
    return new Promise((resolve) => {
        const newImg = new Image();
        
        newImg.onload = () => {
            // Transition plus rapide
            originalImg.style.transition = 'opacity 0.2s ease';
            originalImg.style.opacity = '0.8';
            
            setTimeout(() => {
                originalImg.src = newImg.src;
                originalImg.style.opacity = '1';
                updateOverlay(overlay, '✅ Done!', 'manga-translator-translated');
                
                setTimeout(() => {
                    if (overlay.parentNode) {
                        overlay.remove();
                    }
                    resolve();
                }, 1500);
            }, 200);
        };
        
        newImg.onerror = () => {
            updateOverlay(overlay, '❌ Display error', 'manga-translator-overlay');
            setTimeout(() => {
                if (overlay.parentNode) {
                    overlay.remove();
                }
                resolve();
            }, 2000);
        };
        
        newImg.src = `data:image/png;base64,${result.translated_image_base64}`;
    });
}

// Scan optimisé avec limitation
let scanInProgress = false;
let lastScanTime = 0;

function scanForImagesOptimized() {
    if (!isTranslationActive || scanInProgress) return;
    
    // Throttling : max 1 scan toutes les 2 secondes
    const now = Date.now();
    if (now - lastScanTime < 2000) {
        console.log('🚫 Scan throttled');
        return;
    }
    lastScanTime = now;
    
    scanInProgress = true;
    console.log('🔍 Scan optimisé...');
    
    const images = Array.from(document.querySelectorAll('img'))
        .filter(img => !processedImages.has(img))
        .filter(shouldTranslateImageOptimized)
        .slice(0, 3); // Limite stricte à 3 images simultanées
    
    console.log(`📸 ${images.length} images sélectionnées`);
    
    if (images.length === 0) {
        scanInProgress = false;
        return;
    }
    
    // Traiter les images avec délai
    let processedCount = 0;
    
    const processNext = () => {
        if (processedCount >= images.length) {
            scanInProgress = false;
            console.log('✅ Scan terminé');
            return;
        }
        
        const img = images[processedCount++];
        processedImages.add(img);
        
        processImageOptimized(img)
            .finally(() => {
                // Délai entre images
                setTimeout(processNext, 800);
            });
    };
    
    processNext();
}

// Invalidation du cache des settings quand ils changent
chrome.storage.onChanged.addListener((changes) => {
    if (changes['api-url'] || changes['source-lang'] || changes['target-lang']) {
        cachedSettings = null; // Forcer le rechargement
        console.log('⚙️ Settings cache invalidé');
    }
});
