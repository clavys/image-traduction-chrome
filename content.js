// content.js - Script injecté dans chaque page web - Version optimisée pour images multiples
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

// NOUVELLE GESTION DE FILE D'ATTENTE SÉQUENTIELLE
class TranslationQueue {
    constructor() {
        this.queue = [];
        this.processing = false;
        this.maxConcurrent = 1; // UNE SEULE REQUÊTE À LA FOIS
        this.currentRequests = 0;
        this.stats = {
            processed: 0,
            succeeded: 0,
            failed: 0
        };
    }
    
    add(img) {
        if (this.queue.some(item => item.img === img)) {
            console.log('🔄 Image déjà en file d\'attente');
            return;
        }
        
        this.queue.push({
            img: img,
            timestamp: Date.now(),
            attempts: 0
        });
        
        console.log(`➕ Ajouté en file: ${this.queue.length} images en attente`);
        this.processNext();
    }
    
    async processNext() {
        if (this.processing || this.currentRequests >= this.maxConcurrent || this.queue.length === 0) {
            return;
        }
        
        this.processing = true;
        this.currentRequests++;
        
        const item = this.queue.shift();
        console.log(`⚙️ Traitement de la file: ${this.queue.length} restantes`);
        
        try {
            await this.processItem(item);
            this.stats.succeeded++;
        } catch (error) {
            console.error('❌ Erreur traitement file:', error);
            this.stats.failed++;
            
            // Retry logic
            if (item.attempts < 1) { // Max 1 retry
                item.attempts++;
                this.queue.unshift(item); // Remettre au début
                console.log('🔄 Retry ajouté en file');
            }
        } finally {
            this.currentRequests--;
            this.processing = false;
            
            // Délai entre les traitements pour éviter la surcharge
            setTimeout(() => {
                this.processNext();
            }, 1500); // 1.5 secondes entre chaque image
        }
    }
    
    async processItem(item) {
        const { img } = item;
        
        if (!img || !img.parentNode) {
            throw new Error('Image non valide ou supprimée du DOM');
        }
        
        await processImageSequential(img);
        this.stats.processed++;
    }
    
    clear() {
        this.queue = [];
        console.log('🧹 File d\'attente vidée');
    }
    
    getStats() {
        return {
            ...this.stats,
            queueLength: this.queue.length,
            processing: this.processing
        };
    }
}

const translationQueue = new TranslationQueue();

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
  
  .manga-translator-queued {
    background: rgba(128, 0, 128, 0.8);
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

// CONFIGURATION RÉSEAU SIMPLIFIÉE
const NETWORK_CONFIG = {
    timeout: 25000, // Timeout plus long
    retryDelay: 2000,
    maxRetries: 1 // Moins de retries
};

// Gestionnaire de requêtes simplifié (UNE SEULE À LA FOIS)
let currentController = null;

async function makeSequentialRequest(apiUrl, requestData) {
    // Annuler la requête précédente s'il y en a une
    if (currentController) {
        currentController.abort();
        console.log('🛑 Requête précédente annulée');
    }
    
    currentController = new AbortController();
    const startTime = performance.now();
    
    try {
        console.log('🌐 Nouvelle requête séquentielle...');
        
        const response = await fetch(`${apiUrl}/translate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData),
            signal: currentController.signal
        });
        
        const networkTime = performance.now() - startTime;
        console.log(`⚡ Requête réseau: ${networkTime.toFixed(0)}ms`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        const totalTime = performance.now() - startTime;
        console.log(`📊 Temps total requête: ${totalTime.toFixed(0)}ms`);
        
        return {
            success: true,
            data: result,
            networkTime: networkTime,
            totalTime: totalTime
        };
        
    } catch (error) {
        if (error.name === 'AbortError') {
            throw new Error('Requête annulée');
        }
        throw error;
    } finally {
        if (currentController) {
            currentController = null;
        }
    }
}

// Version séquentielle de la traduction
async function performTranslationSequential(img) {
    try {
        if (!cachedSettings) {
            cachedSettings = await new Promise(resolve => {
                chrome.storage.local.get([
                    'api-url', 'source-lang', 'target-lang'
                ], resolve);
            });
        }
        
        const base64Image = await imageToBase64(img);
        const apiUrl = cachedSettings['api-url'] || 'http://localhost:8000';
        
        const requestData = {
            image_base64: base64Image,
            source_lang: cachedSettings['source-lang'] || 'ja',
            target_lang: cachedSettings['target-lang'] || 'en',
            translator: 'google'
        };
        
        const result = await makeSequentialRequest(apiUrl, requestData);
        console.log(`🚀 Requête réussie en ${result.totalTime.toFixed(0)}ms`);
        
        return result.data;
        
    } catch (error) {
        console.error('Erreur requête séquentielle:', error);
        throw error;
    }
}

// Fonction pour préparer les connexions (simplifiée)
async function warmupConnections() {
    try {
        if (!cachedSettings) {
            cachedSettings = await new Promise(resolve => {
                chrome.storage.local.get(['api-url'], resolve);
            });
        }
        
        const apiUrl = cachedSettings['api-url'] || 'http://localhost:8000';
        
        console.log('🔥 Préchauffage de la connexion...');
        
        const response = await fetch(`${apiUrl}/health`, {
            method: 'GET'
        });
        
        if (response.ok) {
            console.log('✅ Connexion préchauffée');
        }
        
    } catch (error) {
        console.log('⚠️ Préchauffage échoué:', error.message);
    }
}

// Précharger les connexions au démarrage
setTimeout(warmupConnections, 1000);

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
    autoTranslateEnabled = result['auto-translate'] !== false;
    
    if (autoTranslateEnabled) {
      activateTranslation();
      updateButtonDisplay();
    }
  });
}

// Basculer entre les modes
function toggleMode() {
  if (!isTranslationActive && !autoTranslateEnabled) {
    autoTranslateEnabled = true;
    activateTranslation();
  } else if (isTranslationActive && autoTranslateEnabled) {
    autoTranslateEnabled = false;
  } else {
    autoTranslateEnabled = false;
    deactivateTranslation();
  }
  
  updateButtonDisplay();
  
  chrome.storage.local.set({
    [TRANSLATION_KEY]: isTranslationActive,
    'auto-translate': autoTranslateEnabled
  });
}

// Mettre à jour l'affichage du bouton avec stats
function updateButtonDisplay() {
  const button = document.getElementById('manga-translator-btn');
  if (!button) return;
  
  const stats = translationQueue.getStats();
  const queueInfo = stats.queueLength > 0 ? ` (${stats.queueLength})` : '';
  
  if (isTranslationActive && autoTranslateEnabled) {
    button.textContent = `Manga Translator: AUTO${queueInfo}`;
    button.className = 'manga-translator-button auto active';
  } else if (isTranslationActive && !autoTranslateEnabled) {
    button.textContent = `Manga Translator: MANUAL${queueInfo}`;
    button.className = 'manga-translator-button active';
  } else {
    button.textContent = 'Manga Translator: OFF';
    button.className = 'manga-translator-button';
  }
}

// Mettre à jour le bouton périodiquement
setInterval(updateButtonDisplay, 2000);

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
  
  chrome.storage.local.set({[TRANSLATION_KEY]: true});
  
  // Scanner toutes les images avec file d'attente
  scanForImagesWithQueue();
}

function deactivateTranslation() {
  isTranslationActive = false;
  updateButtonDisplay();
  
  // Vider la file d'attente
  translationQueue.clear();
  
  // Annuler la requête en cours
  if (currentController) {
    currentController.abort();
  }
  
  chrome.storage.local.set({[TRANSLATION_KEY]: false});
  
  // Supprimer tous les overlays
  document.querySelectorAll('.manga-translator-overlay').forEach(el => el.remove());
}

// Détecter les changements de page
function detectPageChange() {
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
  
  window.addEventListener('popstate', onPageChange);
  
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
  console.log('🔄 Page changed, resetting translation state...');
  
  // Réinitialiser l'état
  processedImages = new WeakSet();
  translationQueue.clear();
  
  // Annuler les requêtes en cours
  if (currentController) {
    currentController.abort();
  }
  
  // Supprimer les anciens overlays
  document.querySelectorAll('.manga-translator-overlay').forEach(el => el.remove());
  
  // Si auto-traduction activée, démarrer automatiquement
  if (autoTranslateEnabled) {
    console.log('🚀 Auto-translating new page...');
    
    setTimeout(() => {
      if (!isTranslationActive) {
        activateTranslation();
      } else {
        scanForImagesWithQueue();
      }
    }, 2000); // Délai plus long pour laisser les images se charger
  }
}

// Observer les nouvelles images
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
  
  if (autoTranslateEnabled || isTranslationActive) {
    if (!isTranslationActive) {
      activateTranslation();
    }
    
    // Ajouter à la file d'attente au lieu de traiter immédiatement
    setTimeout(() => {
      translationQueue.add(img);
    }, 1000);
  }
}

// Vérifier si une image doit être traduite (version simplifiée)
function shouldTranslateImage(img) {
  const minWidth = 150;
  const minHeight = 80;
  
  if (img.naturalWidth < minWidth || img.naturalHeight < minHeight) {
    return false;
  }
  
  // Éviter les images trop grandes (backgrounds)
  const maxArea = 1500 * 1500;
  if (img.naturalWidth * img.naturalHeight > maxArea) {
    return false;
  }
  
  const skipPatterns = ['logo', 'icon', 'avatar', 'button', 'emoji'];
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
  
  setTimeout(() => {
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

// Traitement séquentiel d'une image
async function processImageSequential(img) {
    try {
        console.log(`🔄 Traitement séquentiel:`, img.src.substring(0, 40) + '...');
        
        const overlay = createOverlay(img, `Processing...`, 'manga-translator-processing');
        
        const startTime = Date.now();
        const result = await performTranslationSequential(img);
        const processingTime = (Date.now() - startTime) / 1000;
        
        if (result.success) {
            await displayTranslatedImage(img, result, overlay);
            
            // Stats
            chrome.runtime.sendMessage({
                action: 'updateStats',
                data: { processingTime: processingTime }
            });
            
            console.log(`✅ Succès en ${processingTime.toFixed(2)}s`);
        } else {
            throw new Error(result.error);
        }
        
    } catch (error) {
        console.warn(`❌ Échec traitement:`, error.message);
        
        const overlay = createOverlay(img, 
            `Failed: ${error.message.substring(0, 15)}...`, 
            'manga-translator-overlay'
        );
        setTimeout(() => overlay.remove(), 4000);
        throw error;
    }
}

// Affichage des résultats traduits
async function displayTranslatedImage(originalImg, result, overlay) {
    return new Promise((resolve) => {
        const newImg = new Image();
        
        newImg.onload = () => {
            originalImg.style.transition = 'opacity 0.3s ease';
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
                }, 2000);
            }, 300);
        };
        
        newImg.onerror = () => {
            updateOverlay(overlay, '❌ Display error', 'manga-translator-overlay');
            setTimeout(() => {
                if (overlay.parentNode) {
                    overlay.remove();
                }
                resolve();
            }, 3000);
        };
        
        newImg.src = `data:image/png;base64,${result.translated_image_base64}`;
    });
}

// Scan avec file d'attente
function scanForImagesWithQueue() {
    if (!isTranslationActive) return;
    
    console.log('🔍 Scan avec file d\'attente...');
    
    const images = Array.from(document.querySelectorAll('img'))
        .filter(img => !processedImages.has(img))
        .filter(shouldTranslateImage)
        .slice(0, 10); // Limite à 6 images maximum
    
    console.log(`📸 ${images.length} images sélectionnées pour la file`);
    
    if (images.length === 0) {
        return;
    }
    
    // Ajouter toutes les images à la file d'attente
    images.forEach((img, index) => {
        processedImages.add(img);
        
        // Overlay temporaire pour indiquer que l'image est en file
        const overlay = createOverlay(img, `En file... (${index + 1})`, 'manga-translator-queued');
        setTimeout(() => {
            if (overlay.parentNode) {
                overlay.remove();
            }
        }, 3000);
        
        // Ajouter à la file avec un petit délai
        setTimeout(() => {
            translationQueue.add(img);
        }, index * 500); // 500ms entre chaque ajout
    });
}

// Cache des settings
let cachedSettings = null;

// Invalidation du cache des settings
chrome.storage.onChanged.addListener((changes) => {
    if (changes['api-url'] || changes['source-lang'] || changes['target-lang']) {
        cachedSettings = null;
        console.log('⚙️ Settings cache invalidé');
    }
});
