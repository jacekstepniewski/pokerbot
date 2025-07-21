import cv2
import os
import numpy as np

# Ścieżki do szablonów (tak jak w głównym kodzie)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RANK_PATHS = [os.path.join(BASE_DIR, "ranks", f) for f in [
    "2.png", "3.png", "4.png", "5.png", "6.png", "7.png", "8.png", "9.png",
    "T.png", "J.png", "Q.png", "K.png", "A.png"
]]
SUIT_PATHS = [os.path.join(BASE_DIR, "suits", f) for f in ["h.png", "d.png", "c.png", "s.png"]]

def load_templates(paths):
    templates = {}
    for path in paths:
        key = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path, 0)
        if img is not None:
            _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            templates[key] = img_bin
    return templates

rank_templates = load_templates(RANK_PATHS)
suit_templates = load_templates(SUIT_PATHS)

def preprocess_for_template(img):
    h, w = img.shape
    return img[0:int(h*0.45), 0:int(w*0.45)]

def best_template_match(region, templates):
    best_name = None
    best_val = 1.0
    for name, tpl in templates.items():
        for scale in np.linspace(0.7, 1.2, 7):
            t_h, t_w = tpl.shape
            sz = (max(1, int(t_w*scale)), max(1, int(t_h*scale)))
            resized = cv2.resize(tpl, sz, interpolation=cv2.INTER_AREA)
            if region.shape[0] < resized.shape[0] or region.shape[1] < resized.shape[1]:
                continue
            res = cv2.matchTemplate(region, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            val = 1 - max_val
            if val < best_val:
                best_val = val
                best_name = name
    return best_name, best_val

def recognize_card(card_img_bgr, rank_templates, suit_templates):
    card_gray = cv2.cvtColor(card_img_bgr, cv2.COLOR_BGR2GRAY)
    corner = preprocess_for_template(card_gray)
    results = []
    for thresh_mode in ['otsu', 'adaptive', 'fixed120']:
        if thresh_mode == 'otsu':
            _, corner_bin = cv2.threshold(corner, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        elif thresh_mode == 'adaptive':
            corner_bin = cv2.adaptiveThreshold(corner, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY_INV, 21, 10)
        elif thresh_mode == 'fixed120':
            _, corner_bin = cv2.threshold(corner, 120, 255, cv2.THRESH_BINARY_INV)
        else:
            continue

        rank_roi = corner_bin[0:int(corner_bin.shape[0]*0.55), 0:int(corner_bin.shape[1]*0.55)]
        rank, r_val = best_template_match(rank_roi, rank_templates)

        sh0 = int(corner_bin.shape[0]*0.55)
        sh1 = int(corner_bin.shape[0]*0.95)
        sw0 = int(corner_bin.shape[1]*0.2)
        sw1 = int(corner_bin.shape[1]*0.8)
        suit_roi = corner_bin[sh0:sh1, sw0:sw1]
        suit, s_val = best_template_match(suit_roi, suit_templates)

        results.append((rank, r_val, suit, s_val))

    results = [res for res in results if res[0] is not None and res[2] is not None]
    if results:
        best = min(results, key=lambda x: x[1]+x[3])
        return best[0]+best[2]
    else:
        return None

if __name__ == "__main__":
    CARD_PATH = os.path.join(BASE_DIR, "c866618d-b6f0-4962-a214-8e36db660abd.png")
    card_img = cv2.imread(CARD_PATH)
    result = recognize_card(card_img, rank_templates, suit_templates)
    print("Rozpoznana karta:", result)
