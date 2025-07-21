import time
import cv2
import numpy as np
import pyautogui
from treys import Card, Evaluator

# Region definitions (x, y, width, height)
FLOP_REGION = (414, 505, 806, 210)
HAND1_REGION = (403, 758, 255, 218)
HAND2_REGION = (973, 758, 255, 218)

# Card template settings
SUITS = ['h', 'd', 'c', 's']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
TEMPLATES = {}

def load_templates(path='templates/'):
    """
    Load card images from the given directory. Templates should be named like 'Ah.png', 'Ts.png', etc.
    """
    for r in RANKS:
        for s in SUITS:
            name = r + s
            img = cv2.imread(f'{path}{name}.png', cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f'Template {path}{name}.png not found')
            TEMPLATES[name] = img


def recognize_card(card_img, threshold=0.8):
    """
    Identify a single card by template matching. Returns the card name like 'Ah' or raises if not found.
    """
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    for name, tpl in TEMPLATES.items():
        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val >= threshold:
            return name
    raise ValueError('Card not recognized')


def get_cards(region, num_cards):
    """
    Capture a region and split it into `num_cards` slices to recognize each card.
    Returns list of card names or raises if any unknown.
    """
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


def wait_for_cards(region, num_cards, timeout=10, poll=0.5):
    """
    Wait until `num_cards` are recognized in the given region or timeout.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            return get_cards(region, num_cards)
        except ValueError:
            time.sleep(poll)
    raise TimeoutError(f'Failed to detect {num_cards} cards in region {region} within {timeout}s')


def card_names_to_treys(names):
    """
    Convert a list of names like ['Ah', 'Td'] to Treys Card objects.
    """
    return [Card.new(n.lower()) for n in names]


def main():
    load_templates()
    evaluator = Evaluator()
    time.sleep(2)  # switch to game window

    for round_idx in range(11):
        # Wait for flop and hands to appear
        flop = wait_for_cards(FLOP_REGION, 5)
        hand1 = wait_for_cards(HAND1_REGION, 2)
        hand2 = wait_for_cards(HAND2_REGION, 2)

        # Evaluate hands
        board = card_names_to_treys(flop)
        h1 = card_names_to_treys(hand1)
        h2 = card_names_to_treys(hand2)
        score1 = evaluator.evaluate(board, h1)
        score2 = evaluator.evaluate(board, h2)

        # Choose better hand (lower score is stronger)
        if score1 < score2:
            click_x, click_y = HAND1_REGION[0] + HAND1_REGION[2] // 2, HAND1_REGION[1] + HAND1_REGION[3] // 2
        else:
            click_x, click_y = HAND2_REGION[0] + HAND2_REGION[2] // 2, HAND2_REGION[1] + HAND2_REGION[3] // 2

        pyautogui.click(click_x, click_y)
        print(f"Round {round_idx+1}: Flop={flop}, Hand1={hand1} (score {score1}), Hand2={hand2} (score {score2}), clicked at ({click_x}, {click_y})")

    print("Finished 11 rounds.")

if __name__ == '__main__':
    main()
