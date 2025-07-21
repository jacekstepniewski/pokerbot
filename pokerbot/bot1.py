import os
import cv2
import numpy as np
from PIL import ImageGrab
from treys import Card, Evaluator

# Absolutna ścieżka do folderu z szablonami
TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Wczytanie wszystkich szablonów (czarno-białe PNG w katalogu TEMPLATE_DIR)
rank_templates = {}
suit_templates = {}

for r in ['2','3','4','5','6','7','8','9','T','J','Q','K','A']:
    path = os.path.join(TEMPLATE_DIR, f'{r}.png')
    tpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        raise FileNotFoundError(f"Rank template not found: {path}")
    rank_templates[r] = tpl

for s in ['h','d','c','s']:
    path = os.path.join(TEMPLATE_DIR, f'{s}.png')
    tpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        raise FileNotFoundError(f"Suit template not found: {path}")
    suit_templates[s] = tpl


def multi_scale_template_match(img, tpl, scale_range=(0.5, 1.5), steps=10):
    """Zwraca (best_val, best_scale, best_loc) dla dopasowania wieloskaliowego między img i tpl."""
    best_val = -1
    best_scale = 1.0
    best_loc = None
    h_img, w_img = img.shape[:2]
    h_tpl_orig, w_tpl_orig = tpl.shape[:2]
    scales = np.linspace(scale_range[0], scale_range[1], steps)
    for scale in scales:
        h_tpl = int(h_tpl_orig * scale)
        w_tpl = int(w_tpl_orig * scale)
        if h_tpl < 5 or w_tpl < 5 or h_tpl > h_img or w_tpl > w_img:
            continue
        tpl_resized = cv2.resize(tpl, (w_tpl, h_tpl), interpolation=cv2.INTER_AREA)
        # zastosuj przejście do binary dla obrazu i szablonu
        res = cv2.matchTemplate(img, tpl_resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = max_val
            best_scale = scale
            best_loc = max_loc
    return best_val, best_scale, best_loc


def match_template(img, templates):
    """Zwraca (klucz, wartość dopasowania) najlepszego template’u przy dopasowaniu wieloskaliowym."""
    best_val = -1
    best_key = None
    for key, tpl in templates.items():
        val, scale, loc = multi_scale_template_match(img, tpl)
        if val > best_val:
            best_val = val
            best_key = key
    return best_key, best_val


def match_with_rotation(img, templates, angles=range(-15, 16, 5)):
    """Próbuj rotować fragment i dopasowywać template’y, zwraca najlepszy przy dopasowaniu wieloskaliowym."""
    best_val = -1
    best_key = None
    for angle in angles:
        h_img, w_img = img.shape[:2]
        M = cv2.getRotationMatrix2D((w_img/2, h_img/2), angle, 1.0)
        rot = cv2.warpAffine(img, M, (w_img, h_img), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        key, val = match_template(rot, templates)
        if val > best_val:
            best_val, best_key = val, key
    return best_key, best_val


def read_card(gray, rank_rect, suit_rect, rotated=False):
    x, y, w, h = rank_rect
    rank_img = gray[y:y+h, x:x+w]
    x2, y2, w2, h2 = suit_rect
    suit_img = gray[y2:y2+h2, x2:x2+w2]
    h_rank, w_rank = rank_img.shape[:2]
    if h_rank == 0 or w_rank == 0:
        raise ValueError(f"Wycinek rank poza ekranem lub o zerowym rozmiarze: {h_rank}x{w_rank}")
    h_suit, w_suit = suit_img.shape[:2]
    if h_suit == 0 or w_suit == 0:
        raise ValueError(f"Wycinek suit poza ekranem lub o zerowym rozmiarze: {h_suit}x{w_suit}")
    if rotated:
        rank, _ = match_with_rotation(rank_img, rank_templates)
        suit, _ = match_with_rotation(suit_img, suit_templates)
    else:
        rank, _ = match_template(rank_img, rank_templates)
        suit, _ = match_template(suit_img, suit_templates)
    card_str = rank + suit
    if len(card_str) != 2:
        raise ValueError(f"Nieprawidłowy kod karty: '{card_str}'")
    return card_str

# Definicje regionów (x, y, szer, wys)
flop_rank = [
    (641, 536, 59, 79),
    (813, 536, 59, 79),
    (990, 536, 59, 79),
    (1165, 536, 59, 79),
    (1340, 536, 59, 79),
]
flop_suit = [
    (641, 620, 48, 68),
    (813, 620, 48, 68),
    (990, 620, 48, 68),
    (1165, 620, 48, 68),
    (1340, 620, 48, 68),
]
hand1_rank = [
    (629, 817, 46, 95),
    (727, 811, 62, 81),
]
hand1_suit = [
    (649, 915, 51, 53),
    (730, 909, 47, 51),
]
hand2_rank = [
    (1249, 797, 62, 105),
    (1340, 803, 58, 80),
]
hand2_suit = [
    (1255, 910, 44, 54),
    (1340, 909, 47, 51),
]

def main():
    screen = np.array(ImageGrab.grab())
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    flop = []
    for idx, (rr, sr) in enumerate(zip(flop_rank, flop_suit), start=1):
        try:
            card = read_card(gray, rr, sr, rotated=False)
        except ValueError as e:
            print(f"Błąd przy czytaniu flop karty #{idx}: {e}")
            raise
        flop.append(card)
    
    hand1 = []
    for idx, (rr, sr) in enumerate(zip(hand1_rank, hand1_suit), start=1):
        try:
            card = read_card(gray, rr, sr, rotated=True)
        except ValueError as e:
            print(f"Błąd przy czytaniu hand1 karty #{idx}: {e}")
            raise
        hand1.append(card)
    
    hand2 = []
    for idx, (rr, sr) in enumerate(zip(hand2_rank, hand2_suit), start=1):
        try:
            card = read_card(gray, rr, sr, rotated=True)
        except ValueError as e:
            print(f"Błąd przy czytaniu hand2 karty #{idx}: {e}")
            raise
        hand2.append(card)

    evaluator = Evaluator()
    try:
        board = [Card.new(c) for c in flop]
    except KeyError as e:
        raise ValueError(f"Nieznany kod w flop: {e}")
    try:
        cards1 = [Card.new(c) for c in hand1]
    except KeyError as e:
        raise ValueError(f"Nieznany kod w hand1: {e}")
    try:
        cards2 = [Card.new(c) for c in hand2]
    except KeyError as e:
        raise ValueError(f"Nieznany kod w hand2: {e}")

    score1 = evaluator.evaluate(board, cards1)
    score2 = evaluator.evaluate(board, cards2)
    chosen = 'hand1' if score1 < score2 else 'hand2'

    print(f'Flop:  {", ".join(flop)}')
    print(f'Hand1: {", ".join(hand1)} → score={score1}')
    print(f'Hand2: {", ".join(hand2)} → score={score2}')
    print(f'\nZalecam wybrać: {chosen}')

if __name__ == '__main__':
    main()
