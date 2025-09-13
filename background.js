// background.js - Service Worker pour l'extension Chrome
console.log('🔧 Manga Translator background script loaded');

// Installation de l'extension
chrome.runtime.onInstalled.addListener((details) => {
  console.log('Extension installée:', details.reason);
  
  // Valeurs par défaut
  const defaultSettings = {
    'manga-translator-active': false,
    'api-url': 'http://localhost:8000',
    'source-lang': 'ja',
    'target-lang': 'en',
    'min-size': 100,
    'stats': {
      translatedCount: 0,
      avgTime: 0,
      lastTranslation: 'Jamais'
    }
  };
  
  chrome.storage.local.set(defaultSettings);
  
  if (details.reason === 'install') {
    // Première installation
    chrome.tabs.create({
      url: chrome.runtime.getURL('popup.html')
    });
  }
});

// Messages entre les scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('Message reçu:', message);
  
  switch (message.action) {
    case 'updateStats':
      updateStats(message.data);
      break;
      
    case 'getStats':
      chrome.storage.local.get(['stats'], (result) => {
        sendResponse(result.stats || {});
      });
      return true; // Réponse asynchrone
      
    case 'resetStats':
      const resetStats = {
        translatedCount: 0,
        avgTime: 0,
        lastTranslation: 'Jamais'
      };
      chrome.storage.local.set({stats: resetStats});
      sendResponse(resetStats);
      break;
      
    default:
      console.log('Action inconnue:', message.action);
  }
});

// Mettre à jour les statistiques
function updateStats(newData) {
  chrome.storage.local.get(['stats'], (result) => {
    const currentStats = result.stats || {
      translatedCount: 0,
      avgTime: 0,
      lastTranslation: 'Jamais'
    };
    
    // Incrémenter le compteur
    currentStats.translatedCount += 1;
    
    // Calculer la moyenne des temps de traitement
    if (newData.processingTime) {
      currentStats.avgTime = (
        (currentStats.avgTime * (currentStats.translatedCount - 1) + newData.processingTime) 
        / currentStats.translatedCount
      );
    }
    
    // Mettre à jour la dernière traduction
    currentStats.lastTranslation = new Date().toLocaleTimeString();
    
    // Sauvegarder
    chrome.storage.local.set({stats: currentStats});
  });
}

// Gestion des erreurs API
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'apiError') {
    console.error('Erreur API:', message.error);
    
    // Afficher une notification si nécessaire
    if (message.critical) {
      chrome.notifications.create({
        type: 'basic',
        iconUrl: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHZpZXdCb3g9IjAgMCA0OCA0OCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjQ4IiBoZWlnaHQ9IjQ4IiByeD0iNiIgZmlsbD0iI2VhNDMzNSIvPgo8dGV4dCB4PSIyNCIgeT0iMzIiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIiBmb250LXNpemU9IjI0IiBmaWxsPSJ3aGl0ZSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+ITwvdGV4dD4KPC9zdmc+Cg==',
        title: 'Manga Translator',
        message: 'API locale inaccessible. Vérifiez que votre serveur est démarré.'
      });
    }
  }
});

// Action du badge de l'extension
chrome.action.onClicked.addListener(async (tab) => {
  // Ouvrir la popup (comportement par défaut)
  // Cette fonction est appelée si pas de popup définie
  console.log('Extension cliquée sur:', tab.url);
});

// Mise à jour du badge selon l'état
chrome.storage.onChanged.addListener((changes) => {
  if (changes['manga-translator-active']) {
    const isActive = changes['manga-translator-active'].newValue;
    
    // Mettre à jour le badge
    chrome.action.setBadgeText({
      text: isActive ? 'ON' : ''
    });
    
    chrome.action.setBadgeBackgroundColor({
      color: isActive ? '#34a853' : '#ea4335'
    });
  }
});

// Initialiser le badge au démarrage
chrome.storage.local.get(['manga-translator-active'], (result) => {
  const isActive = result['manga-translator-active'] || false;
  
  chrome.action.setBadgeText({
    text: isActive ? 'ON' : ''
  });
  
  chrome.action.setBadgeBackgroundColor({
    color: isActive ? '#34a853' : '#ea4335'
  });
});

// Nettoyage périodique du cache (optionnel)
setInterval(() => {
  // Nettoyer les anciennes données si nécessaire
  console.log('Nettoyage périodique...');
}, 60000 * 10); // Toutes les 10 minutes
