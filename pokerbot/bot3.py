import cv2
import os
from treys import Card, Evaluator

# --- Definicje regionów i katalogów szablonów ---
REGIONS = {
    'f1': {'r': (641, 536, 59, 79),  's': (641, 630, 56, 68),  'rank_dir': 'rank',    'suit_dir': 'suit'},
    'f2': {'r': (813, 536, 59, 79),  's': (813, 630, 56, 68),  'rank_dir': 'rank',    'suit_dir': 'suit'},
    'f3': {'r': (990, 536, 59, 79),  's': (990, 630, 56, 68),  'rank_dir': 'rank',    'suit_dir': 'suit'},
    'f4': {'r': (1165, 536, 59, 79), 's': (1165, 630, 56, 68), 'rank_dir': 'rank',    'suit_dir': 'suit'},
    'f5': {'r': (1340, 536, 59, 79), 's': (1340, 630, 56, 68), 'rank_dir': 'rank',    'suit_dir': 'suit'},
    'h1-1': {'r': (635, 816, 66, 76), 's': (649, 915, 51, 53), 'rank_dir': 'rank-h1', 'suit_dir': 'suit-h1'},
    'h1-2': {'r': (733, 808, 59, 79), 's': (729, 909, 47, 51), 'rank_dir': 'rank-h1', 'suit_dir': 'suit-h1'},
    'h2-1': {'r': (1255, 803, 61, 95),'s': (1255, 910, 51, 53),'rank_dir': 'rank-h2', 'suit_dir': 'suit-h2'},
    'h2-2': {'r': (1346, 808, 59, 79),'s': (1331, 900, 52, 59),'rank_dir': 'rank-h2', 'suit_dir': 'suit-h2'},
}

def load_templates_from(dir_path):
    templates = {}
    for fname in os.listdir(dir_path):
        if fname.endswith('.png'):
            key = os.path.splitext(fname)[0]
            img = cv2.imread(os.path.join(dir_path, fname), cv2.IMREAD_GRAYSCALE)
            templates[key] = img
    return templates

def match_template(patch, templates):
    best_key, best_val = None, -1.0
    for key, tpl in templates.items():
        res = cv2.matchTemplate(patch, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = max_val
            best_key = key
    return best_key

def recognize_cards(screen_path='blitz.png'):
    img = cv2.imread(screen_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = {}
    for name, cfg in REGIONS.items():
        x, y, w, h = cfg['r']
        rank_patch = gray[y:y+h, x:x+w]
        x2, y2, w2, h2 = cfg['s']
        suit_patch = gray[y2:y2+h2, x2:x2+w2]

        rank_tpls = load_templates_from(cfg['rank_dir'])
        suit_tpls = load_templates_from(cfg['suit_dir'])

        rank = match_template(rank_patch, rank_tpls)
        suit = match_template(suit_patch, suit_tpls)

        results[name] = rank + suit
    return results

def evaluate_choice(rec):
    evaluator = Evaluator()
    board = [Card.new(rec[f]) for f in ['f1','f2','f3','f4','f5']]
    h1 = [Card.new(rec['h1-1']), Card.new(rec['h1-2'])]
    h2 = [Card.new(rec['h2-1']), Card.new(rec['h2-2'])]

    score1 = evaluator.evaluate(board, h1)
    score2 = evaluator.evaluate(board, h2)
    choice = 'hand1' if score1 < score2 else 'hand2'
    return score1, score2, choice

if __name__ == '__main__':
    rec = recognize_cards('D:\\Python\\hello\\pokerbot\\blitz.png')
    print("Rozpoznane karty:")
    print("Flop:", [rec[f] for f in ['f1','f2','f3','f4','f5']])
    print("Hand1:", rec['h1-1'], rec['h1-2'])
    print("Hand2:", rec['h2-1'], rec['h2-2'])
    s1, s2, best = evaluate_choice(rec)
    print(f"Ocena hand1: {s1}, hand2: {s2}")
    print("Wybrano:", best)
