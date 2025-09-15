// content.js - Script injecté dans chaque page web
console.log('🚀 Manga Translator extension loaded');

// Configuration
const API_BASE_URL = 'http://localhost:8000';
const TRANSLATION_KEY = 'manga-translator-active';

// État de l'extension
let isTranslationActive = false;
let processedImages = new WeakSet();
let autoTranslateEnabled = true; // ⭐ NOUVEAU: Auto-traduction activée

// ⭐ NOUVEAU: Observer les changements de page (SPA, navigation AJAX)
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
  
  // ⭐ NOUVEAU: Charger l'état et démarrer automatiquement
  chrome.storage.local.get([TRANSLATION_KEY, 'auto-translate'], (result) => {
    autoTranslateEnabled = result['auto-translate'] !== false; // Par défaut: true
    
    if (autoTranslateEnabled) {
      activateTranslation();
      updateButtonDisplay();
    }
  });
}

// ⭐ NOUVEAU: Basculer entre les modes
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

// ⭐ NOUVEAU: Mettre à jour l'affichage du bouton
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

// Activer/désactiver la traduction (fonction existante modifiée)
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
  scanForImages();
}

function deactivateTranslation() {
  isTranslationActive = false;
  updateButtonDisplay();
  
  // Sauvegarder l'état
  chrome.storage.local.set({[TRANSLATION_KEY]: false});
  
  // Supprimer tous les overlays
  document.querySelectorAll('.manga-translator-overlay').forEach(el => el.remove());
}

// ⭐ NOUVEAU: Détecter les changements de page
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

// ⭐ NOUVEAU: Actions à effectuer lors d'un changement de page
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
        scanForImages();
      }
    }, 1500); // Délai pour laisser la page se charger
  }
}

// Scanner les images sur la page (fonction existante)
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

// ⭐ NOUVEAU: Observer les nouvelles images avec auto-traduction
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

// ⭐ NOUVEAU: Gérer une nouvelle image détectée
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
    setTimeout(() => processImage(img), 500);
  }
}

// Vérifier si une image doit être traduite (fonction existante)
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

// [... Toutes les autres fonctions existantes restent identiques ...]
// processImage, imageToBase64, createOverlay, etc.

// ⭐ NOUVEAU: Initialisation modifiée
function init() {
  injectStyles();
  createControlButton();
  observeNewImages();
  detectPageChange(); // ⭐ NOUVEAU
  
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

// Message depuis le popup (fonction existante)
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'toggleTranslation') {
    toggleTranslation();
    sendResponse({active: isTranslationActive});
  }
});

// ⭐ NOUVEAU: Les fonctions manquantes (ajoutez-les si elles n'existent pas)

// Traiter une image (version simplifiée - ajoutez votre fonction complète)
async function processImage(img) {
  try {
    const overlay = createOverlay(img, 'Processing...', 'manga-translator-processing');
    const base64Image = await imageToBase64(img);
    
    const response = await fetch(`${API_BASE_URL}/translate`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        image_base64: base64Image,
        source_lang: 'ja',
        target_lang: 'en',
        translator: 'google'
      })
    });
    
    const result = await response.json();
    
    if (result.success) {
      const newImg = new Image();
      newImg.onload = () => {
        img.src = newImg.src;
        updateOverlay(overlay, 'Translated!', 'manga-translator-translated');
        setTimeout(() => overlay.remove(), 2000);
      };
      newImg.src = `data:image/png;base64,${result.translated_image_base64}`;
    } else {
      updateOverlay(overlay, 'Error: ' + result.error, 'manga-translator-overlay');
      setTimeout(() => overlay.remove(), 3000);
    }
  } catch (error) {
    console.error('❌ Translation failed:', error);
  }
}

// Convertir image en base64 (version simplifiée)
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
