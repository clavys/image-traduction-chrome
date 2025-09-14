# AJOUTEZ CETTE MÉTHODE HYBRIDE À VOTRE api_server.py
# Insérez-la AVANT la fonction render_fallback_optimized()

async def render_with_ballons_core_components(img_array, blk_list, mask, request):
    """MÉTHODE HYBRIDE - Utilise les composants internes de BallonsTranslator directement"""
    try:
        print("🔧 Tentative d'accès aux composants internes BallonsTranslator...")
        
        # Essayer d'utiliser les modules internes disponibles
        result_image = None
        
        # 1. Inpainting avec le module natif
        if 'inpainter' in ballons_modules:
            print("🖌️ Inpainting avec module natif...")
            inpainter = ballons_modules['inpainter']
            
            # Créer le masque d'inpainting à partir des TextBlocks
            mask_array = create_smart_inpaint_mask(img_array, blk_list)
            
            # Appliquer l'inpainting natif
            try:
                inpainted_array = inpainter.inpaint(img_array, mask_array)
                print("✅ Inpainting natif réussi")
            except Exception as e:
                print(f"⚠️ Inpainting natif échoué: {e}")
                inpainted_array = img_array.copy()
        else:
            inpainted_array = img_array.copy()
        
        # 2. Rendu du texte avec analyse des propriétés BallonsTranslator
        print("✍️ Rendu texte avec propriétés BallonsTranslator...")
        result_image = render_text_with_ballons_properties(inpainted_array, blk_list)
        
        if result_image is not None:
            print("✅ Rendu hybride réussi")
            return result_image
        
    except Exception as e:
        print(f"❌ Erreur rendu hybride: {e}")
    
    # Fallback vers le rendu optimisé
    return None


def create_smart_inpaint_mask(img_array, blk_list):
    """Créer un masque d'inpainting intelligent basé sur les TextBlocks"""
    mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
    
    for blk in blk_list:
        if not hasattr(blk, 'xyxy') or not blk.translation:
            continue
            
        x1, y1, x2, y2 = map(int, blk.xyxy)
        
        # Contraintes de sécurité
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_array.shape[1], x2)
        y2 = min(img_array.shape[0], y2)
        
        if x2 > x1 and y2 > y1:
            # Expansion intelligente du masque pour meilleur inpainting
            expansion = max(2, min((x2-x1)//10, (y2-y1)//10))
            
            x1_exp = max(0, x1 - expansion)
            y1_exp = max(0, y1 - expansion)
            x2_exp = min(img_array.shape[1], x2 + expansion)
            y2_exp = min(img_array.shape[0], y2 + expansion)
            
            mask[y1_exp:y2_exp, x1_exp:x2_exp] = 255
    
    return mask


def render_text_with_ballons_properties(img_array, blk_list):
    """Rendu de texte utilisant les propriétés extraites par BallonsTranslator"""
    result_image = Image.fromarray(img_array.astype(np.uint8))
    draw = ImageDraw.Draw(result_image)
    
    print(f"🎨 Rendu avec propriétés BallonsTranslator pour {len(blk_list)} blocs")
    
    for i, blk in enumerate(blk_list):
        if not blk.translation or not hasattr(blk, 'xyxy'):
            continue
        
        # Extraire les propriétés du TextBlock BallonsTranslator
        props = extract_ballons_textblock_properties(blk, img_array)
        
        # Utiliser ces propriétés pour un rendu précis
        render_single_textblock_precise(draw, result_image, blk, props, i)
    
    return result_image


def extract_ballons_textblock_properties(blk, img_array):
    """Extraire toutes les propriétés disponibles du TextBlock BallonsTranslator"""
    props = {
        'x1': 0, 'y1': 0, 'x2': 100, 'y2': 50,
        'font_size': 16,
        'orientation': 'horizontal',
        'alignment': 'center',
        'color': (0, 0, 0),
        'background_color': (255, 255, 255),
        'stroke_width': 0,
        'is_bold': False,
        'is_italic': False,
        'language': 'en'
    }
    
    # Coordonnées de base
    if hasattr(blk, 'xyxy'):
        props['x1'], props['y1'], props['x2'], props['y2'] = map(int, blk.xyxy)
    
    # Propriétés de police (BallonsTranslator peut les détecter)
    if hasattr(blk, 'font_size') and blk.font_size > 0:
        props['font_size'] = int(blk.font_size)
    
    if hasattr(blk, 'fontformat'):
        ff = blk.fontformat
        if hasattr(ff, 'font_size') and ff.font_size > 0:
            props['font_size'] = int(ff.font_size)
        if hasattr(ff, 'bold'):
            props['is_bold'] = ff.bold
        if hasattr(ff, 'italic'):
            props['is_italic'] = ff.italic
        if hasattr(ff, 'color'):
            props['color'] = ff.color
        if hasattr(ff, 'stroke_width'):
            props['stroke_width'] = ff.stroke_width
    
    # Orientation (BallonsTranslator peut la détecter)
    if hasattr(blk, 'vertical') and blk.vertical:
        props['orientation'] = 'vertical'
    elif hasattr(blk, 'direction'):
        if 'vertical' in str(blk.direction).lower():
            props['orientation'] = 'vertical'
    
    # Alignement
    if hasattr(blk, 'alignment'):
        align_map = {0: 'left', 1: 'center', 2: 'right'}
        props['alignment'] = align_map.get(blk.alignment, 'center')
    
    # Analyse automatique si pas de propriétés détectées
    zone_width = props['x2'] - props['x1']
    zone_height = props['y2'] - props['y1']
    
    # Auto-détection orientation basée sur dimensions
    if zone_height / zone_width > 2.5:
        props['orientation'] = 'vertical'
    
    # Auto-ajustement taille de police
    if props['font_size'] == 16:  # Valeur par défaut
        if props['orientation'] == 'vertical':
            props['font_size'] = max(10, min(int(zone_width * 0.7), 24))
        else:
            props['font_size'] = max(10, min(int(zone_height * 0.35), 26))
    
    # Analyse couleur de fond pour contraste
    try:
        if (0 <= props['y1'] < img_array.shape[0] and 
            0 <= props['y2'] <= img_array.shape[0] and
            0 <= props['x1'] < img_array.shape[1] and 
            0 <= props['x2'] <= img_array.shape[1]):
            
            region = img_array[props['y1']:props['y2'], props['x1']:props['x2']]
            if region.size > 0:
                avg_brightness = np.mean(region)
                if avg_brightness > 128:
                    props['color'] = (0, 0, 0)
                    props['background_color'] = (255, 255, 255)
                else:
                    props['color'] = (255, 255, 255)
                    props['background_color'] = (0, 0, 0)
    except:
        pass
    
    return props


def render_single_textblock_precise(draw, result_image, blk, props, index):
    """Rendu précis d'un seul TextBlock avec toutes les propriétés"""
    
    translation = blk.translation
    x1, y1, x2, y2 = props['x1'], props['y1'], props['x2'], props['y2']
    zone_width = x2 - x1
    zone_height = y2 - y1
    
    print(f"  📝 TextBlock {index+1}: '{translation[:20]}...' ({zone_width}x{zone_height})")
    
    # Chargement de police avec propriétés
    font = load_font_with_properties(props)
    
    # Préparation des lignes selon l'orientation
    lines = prepare_text_lines(translation, props, draw, font, zone_width)
    
    # Calcul du positionnement
    positions = calculate_text_positions(lines, props, draw, font, zone_width, zone_height)
    
    # Fond semi-transparent intelligent
    draw_smart_background(draw, result_image, props, positions, len(lines))
    
    # Rendu du texte avec toutes les propriétés
    draw_text_with_properties(draw, lines, positions, props, font)


def load_font_with_properties(props):
    """Charger une police avec toutes les propriétés (gras, italique, etc.)"""
    size = props['font_size']
    
    # Essayer de charger des polices avec variations
    font_variants = []
    
    if props['is_bold'] and props['is_italic']:
        font_variants = ["arialbi.ttf", "calibriz.ttf", "segoeuib.ttf"]
    elif props['is_bold']:
        font_variants = ["arialbd.ttf", "calibrib.ttf", "segoeuib.ttf"]
    elif props['is_italic']:
        font_variants = ["ariali.ttf", "calibrii.ttf", "segoeuii.ttf"]
    else:
        font_variants = ["arial.ttf", "calibri.ttf", "segoeui.ttf"]
    
    # Ajouter les polices de base
    font_variants.extend(["arial.ttf", "calibri.ttf", "tahoma.ttf"])
    
    for font_name in font_variants:
        try:
            return ImageFont.truetype(font_name, size)
        except:
            continue
    
    return ImageFont.load_default()


def prepare_text_lines(text, props, draw, font, zone_width):
    """Préparer les lignes de texte selon l'orientation et les propriétés"""
    
    if props['orientation'] == 'vertical':
        # Mode vertical : chaque caractère est une ligne
        lines = [char for char in text if char.strip()]
    else:
        # Mode horizontal : découpage intelligent par mots
        words = text.split()
        lines = []
        current_line = ""
        margin = max(8, zone_width * 0.08)
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            text_width = draw.textlength(test_line, font=font)
            
            if text_width <= zone_width - margin:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Mot trop long : le raccourcir
                    max_chars = max(1, (zone_width - margin) // (props['font_size'] // 2))
                    lines.append(word[:max_chars])
        
        if current_line:
            lines.append(current_line)
    
    return lines


def calculate_text_positions(lines, props, draw, font, zone_width, zone_height):
    """Calculer les positions précises pour chaque ligne"""
    positions = []
    
    line_height = props['font_size'] + (3 if props['orientation'] == 'vertical' else 5)
    total_height = len(lines) * line_height
    
    # Position Y de départ
    if total_height <= zone_height:
        start_y = props['y1'] + (zone_height - total_height) // 2
    else:
        start_y = props['y1'] + 3
    
    for i, line in enumerate(lines):
        if props['orientation'] == 'vertical':
            # Centrage horizontal pour texte vertical
            text_x = props['x1'] + (zone_width - props['font_size']) // 2
            text_y = start_y + i * line_height
        else:
            # Alignement selon la propriété
            text_width = draw.textlength(line, font=font)
            
            if props['alignment'] == 'left':
                text_x = props['x1'] + 5
            elif props['alignment'] == 'right':
                text_x = props['x2'] - text_width - 5
            else:  # center
                text_x = props['x1'] + (zone_width - text_width) // 2
            
            text_y = start_y + i * line_height
        
        # Contraintes de sécurité
        text_x = max(2, min(text_x, props['x2'] - 10))
        text_y = max(2, min(text_y, props['y2'] - props['font_size'] - 2))
        
        positions.append((text_x, text_y))
    
    return positions


def draw_smart_background(draw, result_image, props, positions, num_lines):
    """Dessiner un fond intelligent pour la lisibilité"""
    if not positions:
        return
    
    # Calculer les limites du fond
    min_x = min(pos[0] for pos in positions) - 3
    min_y = min(pos[1] for pos in positions) - 2
    max_x = max(pos[0] for pos in positions) + 100  # Approximation largeur
    max_y = max(pos[1] for pos in positions) + props['font_size'] + 2
    
    # Ajuster au TextBlock
    min_x = max(props['x1'], min_x)
    min_y = max(props['y1'], min_y)
    max_x = min(props['x2'], max_x)
    max_y = min(props['y2'], max_y)
    
    # Fond semi-transparent
    overlay = Image.new('RGBA', result_image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    bg_color = props['background_color'] + (200,)  # Ajouter transparence
    
    overlay_draw.rounded_rectangle(
        [min_x, min_y, max_x, max_y],
        radius=max(2, props['font_size'] // 8),
        fill=bg_color
    )
    
    # Composer avec l'image
    result_image.paste(
        Image.alpha_composite(result_image.convert('RGBA'), overlay).convert('RGB')
    )


def draw_text_with_properties(draw, lines, positions, props, font):
    """Dessiner le texte avec toutes les propriétés (ombre, contour, etc.)"""
    
    text_color = props['color']
    stroke_width = props['stroke_width']
    
    for line, (text_x, text_y) in zip(lines, positions):
        # Ombre subtile si pas de contour
        if stroke_width == 0:
            shadow_offset = max(1, props['font_size'] // 15)
            shadow_color = (128, 128, 128) if text_color == (0, 0, 0) else (64, 64, 64)
            
            draw.text(
                (text_x + shadow_offset, text_y + shadow_offset),
                line,
                fill=shadow_color,
                font=font
            )
        
        # Texte principal avec contour si spécifié
        if stroke_width > 0:
            # Contour
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text(
                            (text_x + dx, text_y + dy),
                            line,
                            fill=(255, 255, 255),  # Contour blanc
                            font=font
                        )
        
        # Texte principal
        draw.text((text_x, text_y), line, fill=text_color, font=font)


# MODIFIEZ AUSSI VOTRE FONCTION render_with_ballons_perfect_method()
# Ajoutez cette méthode en PREMIÈRE position :

async def render_with_ballons_perfect_method_updated(img_array, blk_list, mask, request):
    """LA MÉTHODE PARFAITE - Version mise à jour avec composants hybrides"""
    
    # Essayer les méthodes dans l'ordre de priorité (NOUVEAU : méthode hybride en premier)
    methods = [
        ("Composants Hybrides BallonsTranslator", render_with_ballons_core_components),
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
