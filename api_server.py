# RENDU FALLBACK ULTRA-OPTIMISÉ - Remplacez votre fonction render_fallback_optimized()

def render_fallback_optimized(image, blk_list):
    """Rendu fallback ULTRA-optimisé - Qualité quasi-native BallonsTranslator"""
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    
    print("🎨 Utilisation du rendu fallback ULTRA-optimisé")
    
    # Traiter chaque TextBlock avec la précision BallonsTranslator
    for i, blk in enumerate(blk_list):
        if not blk.translation or not hasattr(blk, 'xyxy'):
            continue
            
        x1, y1, x2, y2 = map(int, blk.xyxy)
        zone_width = x2 - x1
        zone_height = y2 - y1
        
        print(f"📝 Rendu TextBlock {i+1}: zone {zone_width}x{zone_height}")
        
        # === DÉTECTION AUTOMATIQUE DES PROPRIÉTÉS MANGA ===
        
        # 1. Détection de l'orientation (vertical pour manga japonais)
        aspect_ratio = zone_height / zone_width if zone_width > 0 else 1
        is_vertical = aspect_ratio > 2.0  # Zone haute et étroite = vertical
        
        # 2. Analyse du texte pour déterminer la langue et l'orientation
        translation = blk.translation
        has_cjk = any('\u4e00' <= char <= '\u9fff' or '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in translation)
        
        # Force vertical si détecté comme manga japonais ou si zone très haute
        if has_cjk or aspect_ratio > 3.0:
            is_vertical = True
        
        # 3. Calcul automatique de la taille de police (méthode BallonsTranslator)
        # Utiliser les propriétés détectées par BallonsTranslator si disponibles
        base_font_size = 16
        if hasattr(blk, 'font_size') and blk.font_size > 0:
            base_font_size = int(blk.font_size)
        elif hasattr(blk, 'fontformat') and hasattr(blk.fontformat, 'font_size') and blk.fontformat.font_size > 0:
            base_font_size = int(blk.fontformat.font_size)
        
        # Adaptation automatique selon la zone (algorithme BallonsTranslator)
        if is_vertical:
            # Pour texte vertical : taille basée sur la largeur
            auto_font_size = min(zone_width * 0.8, zone_height / len(translation) * 0.9)
        else:
            # Pour texte horizontal : taille basée sur la hauteur
            auto_font_size = min(zone_height * 0.4, zone_width / len(translation) * 1.2)
        
        # Combinaison intelligente
        font_size = max(10, min(base_font_size, int(auto_font_size), 32))
        
        # 4. Chargement de police optimisée
        font = get_optimized_font(font_size)
        
        # === PRÉPARATION DU TEXTE SELON L'ORIENTATION ===
        
        if is_vertical:
            # TEXTE VERTICAL (style manga japonais)
            print(f"    📜 Mode vertical détecté")
            
            # Chaque caractère sur une ligne (style authentique manga)
            lines = []
            for char in translation:
                if char.strip():  # Ignorer les espaces
                    lines.append(char)
            
            # Limiter selon la hauteur disponible
            max_chars = max(1, zone_height // (font_size + 2))
            if len(lines) > max_chars:
                lines = lines[:max_chars-1] + ['…']
                
        else:
            # TEXTE HORIZONTAL (style comics occidentaux)
            print(f"    📄 Mode horizontal détecté")
            
            # Découpage intelligent par mots avec mesure précise
            words = translation.split()
            lines = []
            current_line = ""
            
            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                text_width = draw.textlength(test_line, font=font)
                
                # Marge adaptative selon la taille de la zone
                margin = max(10, zone_width * 0.1)
                
                if text_width <= zone_width - margin:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                        current_line = word
                    else:
                        # Mot trop long : le découper
                        lines.append(word[:zone_width // (font_size // 2)])
                        
            if current_line:
                lines.append(current_line)
            
            # Limiter selon la hauteur disponible
            line_height = font_size + 4
            max_lines = max(1, zone_height // line_height)
            if len(lines) > max_lines:
                lines = lines[:max_lines-1] + [lines[max_lines-1][:8] + "…"]
        
        # === POSITIONNEMENT PRÉCIS ===
        
        line_height = font_size + (2 if is_vertical else 4)
        total_text_height = len(lines) * line_height
        
        # Centrage vertical intelligent
        if total_text_height <= zone_height:
            start_y = y1 + (zone_height - total_text_height) // 2
        else:
            start_y = y1 + 5  # Coller en haut si trop grand
        
        # === FOND ADAPTATIF POUR LISIBILITÉ ===
        
        # Analyser la couleur de fond moyenne dans la zone
        region = np.array(result_image)[y1:y2, x1:x2]
        avg_brightness = np.mean(region) if region.size > 0 else 128
        
        # Choisir la couleur de fond et de texte selon le contraste
        if avg_brightness > 128:
            # Fond clair -> texte noir, fond blanc
            text_color = (0, 0, 0)
            bg_color = (255, 255, 255, 200)
        else:
            # Fond sombre -> texte blanc, fond noir
            text_color = (255, 255, 255)
            bg_color = (0, 0, 0, 200)
        
        # Créer le fond avec bordures arrondies
        bg_margin = max(3, font_size // 4)
        bg_x1 = max(0, x1 - bg_margin)
        bg_y1 = max(0, start_y - bg_margin)
        bg_x2 = min(result_image.width, x2 + bg_margin)
        bg_y2 = min(result_image.height, start_y + total_text_height + bg_margin)
        
        # Fond semi-transparent avec bordures douces
        overlay = Image.new('RGBA', result_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rounded_rectangle(
            [bg_x1, bg_y1, bg_x2, bg_y2], 
            radius=max(2, font_size // 6),
            fill=bg_color
        )
        
        # Composer avec l'image principale
        result_image = Image.alpha_composite(
            result_image.convert('RGBA'), 
            overlay
        ).convert('RGB')
        
        # Redessiner sur l'image composée
        draw = ImageDraw.Draw(result_image)
        
        # === RENDU DU TEXTE LIGNE PAR LIGNE ===
        
        for line_idx, line in enumerate(lines):
            if is_vertical:
                # POSITIONNEMENT VERTICAL (style manga)
                text_x = x1 + (zone_width - font_size) // 2
                text_y = start_y + line_idx * line_height
                
                # Ajustement fin pour centrage parfait
                text_x = max(5, min(text_x, result_image.width - font_size - 5))
                
            else:
                # POSITIONNEMENT HORIZONTAL (style comics)
                text_width = draw.textlength(line, font=font)
                
                # Centrage horizontal précis
                text_x = x1 + (zone_width - text_width) // 2
                text_y = start_y + line_idx * line_height
                
                # Contraintes des limites
                text_x = max(5, min(text_x, result_image.width - text_width - 5))
            
            # Contraintes verticales
            text_y = max(5, min(text_y, result_image.height - font_size - 5))
            
            # Vérifier si on dépasse les limites
            if text_y + font_size > result_image.height:
                break
            
            # RENDU AVEC OMBRAGE SUBTIL (style professionnel)
            
            # Ombre légère pour profondeur
            shadow_offset = max(1, font_size // 12)
            shadow_color = (128, 128, 128) if avg_brightness > 128 else (64, 64, 64)
            
            draw.text(
                (text_x + shadow_offset, text_y + shadow_offset), 
                line, 
                fill=shadow_color, 
                font=font
            )
            
            # Texte principal
            draw.text((text_x, text_y), line, fill=text_color, font=font)
            
            print(f"    ✏️ Ligne {line_idx+1}: '{line}' à ({text_x}, {text_y})")
    
    # === SIGNATURE DISCRÈTE ===
    small_font = get_optimized_font(8)
    signature_text = "BallonsTranslator Enhanced"
    sig_width = draw.textlength(signature_text, font=small_font)
    
    draw.text(
        (result_image.width - sig_width - 5, result_image.height - 15), 
        signature_text, 
        fill=(120, 120, 120, 180), 
        font=small_font
    )
    
    print(f"✅ Rendu fallback terminé - {len([b for b in blk_list if b.translation])} TextBlocks rendus")
    
    return result_image


def get_optimized_font(size):
    """Chargement de police optimisée avec fallback intelligent"""
    
    # Liste de polices par ordre de préférence (meilleure qualité)
    font_preferences = [
        # Polices Windows haute qualité
        "segoeui.ttf",      # Segoe UI - Excellent pour l'affichage
        "calibri.ttf",      # Calibri - Très lisible
        "arial.ttf",        # Arial - Standard
        "tahoma.ttf",       # Tahoma - Compact et net
        "verdana.ttf",      # Verdana - Très lisible
        
        # Polices Mac
        "Helvetica.ttc",
        "Arial.ttf",
        
        # Polices Linux
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf"
    ]
    
    for font_name in font_preferences:
        try:
            font = ImageFont.truetype(font_name, size)
            return font
        except:
            continue
    
    # Fallback vers police par défaut
    try:
        return ImageFont.load_default()
    except:
        # Dernier recours : police PIL basique
        return None


# === FONCTION D'ANALYSE AVANCÉE (BONUS) ===

def analyze_textblock_properties(blk, img_array):
    """Analyse avancée des propriétés d'un TextBlock pour un rendu optimal"""
    properties = {
        'orientation': 'horizontal',
        'font_size': 16,
        'text_color': (0, 0, 0),
        'background_brightness': 128
    }
    
    if not hasattr(blk, 'xyxy'):
        return properties
    
    x1, y1, x2, y2 = map(int, blk.xyxy)
    zone_width = x2 - x1
    zone_height = y2 - y1
    
    # Analyse de l'orientation basée sur les dimensions
    aspect_ratio = zone_height / zone_width if zone_width > 0 else 1
    if aspect_ratio > 2.5:
        properties['orientation'] = 'vertical'
    
    # Extraction de la zone pour analyse
    try:
        if (0 <= y1 < img_array.shape[0] and 0 <= y2 <= img_array.shape[0] and 
            0 <= x1 < img_array.shape[1] and 0 <= x2 <= img_array.shape[1]):
            
            region = img_array[y1:y2, x1:x2]
            
            # Analyse de la luminosité moyenne
            if region.size > 0:
                properties['background_brightness'] = float(np.mean(region))
            
            # Estimation de la taille de police basée sur la zone
            if properties['orientation'] == 'vertical':
                properties['font_size'] = max(10, min(int(zone_width * 0.7), 24))
            else:
                properties['font_size'] = max(10, min(int(zone_height * 0.4), 28))
                
    except Exception as e:
        print(f"⚠️ Erreur analyse TextBlock: {e}")
    
    return properties
