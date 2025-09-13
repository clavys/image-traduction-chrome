// popup.js - Logique de la popup
document.addEventListener('DOMContentLoaded', function() {
    // Éléments DOM
    const statusDiv = document.getElementById('status');
    const toggleBtn = document.getElementById('toggle-btn');
    const scanBtn = document.getElementById('scan-btn');
    const testApiBtn = document.getElementById('test-api-btn');
    const apiStatusDot = document.getElementById('api-status-dot');
    const apiStatusText = document.getElementById('api-status-text');
    const statsDiv = document.getElementById('stats');
    
    // Inputs des paramètres
    const apiUrlInput = document.getElementById('api-url');
    const sourceLangSelect = document.getElementById('source-lang');
    const targetLangSelect = document.getElementById('target-lang');
    const minSizeInput = document.getElementById('min-size');
    
    // État initial
    let isActive = false;
    let apiOnline = false;
    
    // Charger les paramètres sauvegardés
    loadSettings();
    
    // Vérifier l'état de l'API
    checkApiStatus();
    
    // Event listeners
    toggleBtn.addEventListener('click', toggleTranslation);
    scanBtn.addEventListener('click', scanCurrentPage);
    testApiBtn.addEventListener('click', testApi);
    
    // Sauvegarder les paramètres quand ils changent
    [apiUrlInput, sourceLangSelect, targetLangSelect, minSizeInput].forEach(input => {
        input.addEventListener('change', saveSettings);
    });
    
    // Charger les paramètres depuis le storage
    function loadSettings() {
        chrome.storage.local.get([
            'manga-translator-active',
            'api-url',
            'source-lang',
            'target-lang',
            'min-size',
            'stats'
        ], (result) => {
            // État actif
            isActive = result['manga-translator-active'] || false;
            updateUI();
            
            // Paramètres
            if (result['api-url']) {
                apiUrlInput.value = result['api-url'];
            }
            if (result['source-lang']) {
                sourceLangSelect.value = result['source-lang'];
            }
            if (result['target-lang']) {
                targetLangSelect.value = result['target-lang'];
            }
            if (result['min-size']) {
                minSizeInput.value = result['min-size'];
            }
            
            // Statistiques
            if (result.stats) {
                updateStats(result.stats);
            }
        });
    }
    
    // Sauvegarder les paramètres
    function saveSettings() {
        const settings = {
            'api-url': apiUrlInput.value,
            'source-lang': sourceLangSelect.value,
            'target-lang': targetLangSelect.value,
            'min-size': parseInt(minSizeInput.value)
        };
        
        chrome.storage.local.set(settings);
    }
    
    // Basculer la traduction
    async function toggleTranslation() {
        if (!apiOnline) {
            showMessage('API hors ligne! Vérifiez que votre serveur local tourne.', 'error');
            return;
        }
        
        try {
            // Obtenir l'onglet actuel
            const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
            
            // Envoyer message au content script
            const response = await chrome.tabs.sendMessage(tab.id, {
                action: 'toggleTranslation'
            });
            
            isActive = response.active;
            updateUI();
            
            showMessage(isActive ? 'Traduction activée!' : 'Traduction désactivée!', 'success');
            
        } catch (error) {
            console.error('Erreur toggle:', error);
            showMessage('Erreur: Rechargez la page et réessayez.', 'error');
        }
    }
    
    // Scanner la page actuelle
    async function scanCurrentPage() {
        if (!apiOnline) {
            showMessage('API hors ligne!', 'error');
            return;
        }
        
        try {
            const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
            
            await chrome.tabs.sendMessage(tab.id, {
                action: 'scanImages'
            });
            
            showMessage('Scan en cours...', 'info');
            
        } catch (error) {
            console.error('Erreur scan:', error);
            showMessage('Erreur de scan.', 'error');
        }
    }
    
    // Tester l'API
    async function testApi() {
        const apiUrl = apiUrlInput.value;
        testApiBtn.textContent = 'Test en cours...';
        testApiBtn.disabled = true;
        
        try {
            // Image de test simple (pixel 1x1)
            const testImage = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==';
            
            const response = await fetch(`${apiUrl}/translate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_base64: testImage,
                    source_lang: sourceLangSelect.value,
                    target_lang: targetLangSelect.value
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                showMessage(`✅ API OK! (${result.processing_time.toFixed(2)}s)`, 'success');
                apiOnline = true;
                updateApiStatus();
            } else {
                showMessage(`❌ Erreur API: ${result.error}`, 'error');
                apiOnline = false;
                updateApiStatus();
            }
            
        } catch (error) {
            console.error('Test API failed:', error);
            showMessage(`❌ API inaccessible: ${error.message}`, 'error');
            apiOnline = false;
            updateApiStatus();
        }
        
        testApiBtn.textContent = 'Tester l\'API';
        testApiBtn.disabled = false;
    }
    
    // Vérifier le statut de l'API
    async function checkApiStatus() {
        const apiUrl = apiUrlInput.value;
        
        try {
            const response = await fetch(`${apiUrl}/health`);
            const result = await response.json();
            
            if (result.status === 'healthy' || result.status === 'simulation_mode') {
                apiOnline = true;
                apiStatusText.textContent = `API: ${result.status}`;
            } else {
                apiOnline = false;
                apiStatusText.textContent = 'API: Erreur';
            }
            
        } catch (error) {
            apiOnline = false;
            apiStatusText.textContent = 'API: Hors ligne';
        }
        
        updateApiStatus();
    }
    
    // Mettre à jour l'interface
    function updateUI() {
        if (isActive) {
            statusDiv.textContent = 'Extension: Activée';
            statusDiv.className = 'status active';
            toggleBtn.textContent = 'Désactiver la traduction';
            toggleBtn.style.background = '#ea4335';
            statsDiv.style.display = 'block';
        } else {
            statusDiv.textContent = 'Extension: Désactivée';
            statusDiv.className = 'status inactive';
            toggleBtn.textContent = 'Activer la traduction';
            toggleBtn.style.background = '#4285f4';
            statsDiv.style.display = 'none';
        }
    }
    
    // Mettre à jour le statut API
    function updateApiStatus() {
        if (apiOnline) {
            apiStatusDot.className = 'status-dot online';
        } else {
            apiStatusDot.className = 'status-dot offline';
        }
    }
    
    // Afficher un message temporaire
    function showMessage(text, type) {
        // Créer un élément de message temporaire
        const message = document.createElement('div');
        message.textContent = text;
        message.style.cssText = `
            position: fixed;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            background: ${type === 'success' ? '#34a853' : type === 'error' ? '#ea4335' : '#4285f4'};
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            z-index: 10000;
        `;
        
        document.body.appendChild(message);
        
        // Supprimer après 3 secondes
        setTimeout(() => {
            if (message.parentNode) {
                message.parentNode.removeChild(message);
            }
        }, 3000);
    }
    
    // Mettre à jour les statistiques
    function updateStats(stats) {
        if (stats) {
            document.getElementById('translated-count').textContent = stats.translatedCount || 0;
            document.getElementById('avg-time').textContent = (stats.avgTime || 0).toFixed(2) + 's';
            document.getElementById('last-translation').textContent = stats.lastTranslation || 'Jamais';
        }
    }
    
    // Rafraîchir les statistiques depuis le storage
    function refreshStats() {
        chrome.storage.local.get(['stats'], (result) => {
            if (result.stats) {
                updateStats(result.stats);
            }
        });
    }
    
    // Rafraîchir les stats toutes les 2 secondes si actif
    setInterval(() => {
        if (isActive) {
            refreshStats();
        }
    }, 2000);
});
