import time
import cv2
import numpy as np
import pyautogui
from treys import Card, Evaluator
import os

# Region definitions (x, y, width, height)
FLOP_REGION = (414, 505, 806, 210)
HAND1_REGION = (403, 758, 255, 218)
HAND2_REGION = (973, 758, 255, 218)

# Template settings
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['h', 'd', 'c', 's']
RANK_TEMPLATES = {}
SUIT_TEMPLATES = {}

# Relative ROI for rank and suit within a card image (x, y, w, h) as fractions
RANK_ROI = (0.02, 0.02, 0.2, 0.2)
SUIT_ROI = (0.02, 0.6, 0.2, 0.3)

absolute_path = os.path.dirname(os.path.abspath(__file__))


def load_templates(path = absolute_path + '\\templates\\'):
    """
    Load separate rank and suit templates from given directories:
    templates/ranks/{rank}.png and templates/suits/{suit}.png
    """
    for r in RANKS:
        img = cv2.imread(f'{path}ranks\\{r}.png', cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f'Rank template {path}ranks\\{r}.png not found')
        RANK_TEMPLATES[r] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for s in SUITS:
        img = cv2.imread(f'{path}suits\\{s}.png', cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f'Suit template {path}suits\\{s}.png not found')
        SUIT_TEMPLATES[s] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

Is this weel random? :-)
def match_template(gray_roi, templates, threshold=0.7):
    """
    Return key of best matching template above threshold.
    """
    best, best_val = None, threshold
    for name, tpl in templates.items():
        res = cv2.matchTemplate(gray_roi, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_val:
            best, best_val = name, max_val
    return best


def recognize_card(card_img):
    """
    Identify rank and suit separately and return combined name like 'Ah'.
    """
    h, w = card_img.shape[:2]
    # Extract rank region
    x, y, rw, rh = RANK_ROI
    rx, ry = int(x*w), int(y*h)
    rw_px, rh_px = int(rw*w), int(rh*h)
    rank_img = cv2.cvtColor(card_img[ry:ry+rh_px, rx:rx+rw_px], cv2.COLOR_BGR2GRAY)
    # Extract suit region
    x, y, sw, sh = SUIT_ROI
    sx, sy = int(x*w), int(y*h)
    sw_px, sh_px = int(sw*w), int(sh*h)
    suit_img = cv2.cvtColor(card_img[sy:sy+sh_px, sx:sx+sw_px], cv2.COLOR_BGR2GRAY)
    # Match
    rank = match_template(rank_img, RANK_TEMPLATES)
    suit = match_template(suit_img, SUIT_TEMPLATES)
    if not rank or not suit:
        raise ValueError('Rank or suit not recognized')
    return rank + suit


def get_cards(region, num_cards):
    x, y, w, h = region
    shot = pyautogui.screenshot(region=(x, y, w, h))
    img = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
    card_width = w // num_cards
    cards = []
    for i in range(num_cards):
        cx = i * card_width
        card_img = img[0:h, cx:cx + card_width]
        name = recognize_card(card_img)
        cards.append(name)
    return cards


def capture_region(region):
    x, y, w, h = region
    shot = pyautogui.screenshot(region=(x, y, w, h))
    return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)


def wait_for_new_deal(prev_img, region, threshold=100000, timeout=10, poll=0.5):
    start = time.time()
    while time.time() - start < timeout:
        curr = capture_region(region)
        diff = cv2.absdiff(prev_img, curr)
        non_zero = np.sum(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY))
        if non_zero > threshold:
            return curr
        time.sleep(poll)
    raise TimeoutError('No new deal detected within timeout')


def card_names_to_treys(names):
    return [Card.new(n.lower()) for n in names]


def main():
    load_templates()
    evaluator = Evaluator()
    time.sleep(2)  # switch to game window

    # Initial capture
    prev_flop = capture_region(FLOP_REGION)
    prev_hand1 = capture_region(HAND1_REGION)
    prev_hand2 = capture_region(HAND2_REGION)

    for round_idx in range(11):
        try:
            new_flop = wait_for_new_deal(prev_flop, FLOP_REGION)
            new_hand1 = wait_for_new_deal(prev_hand1, HAND1_REGION)
            new_hand2 = wait_for_new_deal(prev_hand2, HAND2_REGION)
        except TimeoutError as e:
            print(e)
            break
        prev_flop, prev_hand1, prev_hand2 = new_flop, new_hand1, new_hand2

        flop = get_cards(FLOP_REGION, 5)
        hand1 = get_cards(HAND1_REGION, 2)
        hand2 = get_cards(HAND2_REGION, 2)

        board = card_names_to_treys(flop)
        h1 = card_names_to_treys(hand1)
        h2 = card_names_to_treys(hand2)
        score1 = evaluator.evaluate(board, h1)
        score2 = evaluator.evaluate(board, h2)

        if score1 < score2:
            region = HAND1_REGION
        else:
            region = HAND2_REGION
        cx = region[0] + region[2]//2
        cy = region[1] + region[3]//2
        pyautogui.click(cx, cy)

        print(f"Round {round_idx+1}: Flop={flop}, Hand1={hand1}, Hand2={hand2}, clicked at ({cx},{cy})")

    print("Finished or stopped.")

if __name__ == '__main__':
    main()
