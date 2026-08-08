"""Read the public game-state bar at the top of the screen.

The bar shows how many cards of each rank are still in play
(WONDER 3 4 5 6 7 8 9 10 J Q K A 2 JOKER). The bot will eventually
track this itself from observed plays, so this reader exists purely as
a verification tool to check that the tracked state matches the game.

The count digits are the only warm-colored (orange/red/cream) marks in
the bar, so they are segmented by color and classified against small
digit templates extracted from a reference screenshot.
"""

import cv2
import numpy as np
from pathlib import Path

from GameLogic.Card import Rank

PROJECT_ROOT = Path(__file__).parent
DIGIT_TEMPLATE_DIR = PROJECT_ROOT / 'Image' / 'templates' / 'digits'

# Left-to-right order of the columns in the bar.
RANK_ORDER = [
    Rank.WONDER, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN,
    Rank.EIGHT, Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING,
    Rank.ACE, Rank.TWO, Rank.JOKER,
]

# Bar location as fractions of the full frame (measured on 1920x1080).
BAR_REGION = {'top': 0.028, 'bottom': 0.111, 'left': 0.334, 'right': 0.782}

# Canonical size digits are normalized to before comparison.
DIGIT_SIZE = (24, 30)


class StatusBarReader:
    def __init__(self, template_dir=DIGIT_TEMPLATE_DIR):
        self.templates = {}
        for template_file in Path(template_dir).glob('*.png'):
            digit = int(template_file.stem)
            image = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            self.templates[digit] = cv2.resize(image, DIGIT_SIZE)
        if not self.templates:
            raise FileNotFoundError(f"No digit templates in {template_dir}")

    @staticmethod
    def _digit_mask(bar):
        """Isolate the warm-colored count digits from the dark bar."""
        b, g, r = cv2.split(bar.astype(np.int16))
        return ((r - b > 30) & (r > 100)).astype(np.uint8) * 255

    @staticmethod
    def _find_digit_blobs(mask):
        """Connected components that look like count digits, left to right."""
        # Blob size limits are relative to the frame so other
        # resolutions keep working (values measured at 1080p).
        scale = mask.shape[1] / 860
        min_h, max_h = 15 * scale, 50 * scale
        max_w = 45 * scale
        min_area = 50 * scale * scale

        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        blobs = []
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if area >= min_area and min_h <= h <= max_h and w <= max_w:
                blobs.append((x, y, w, h))
        blobs.sort()
        return blobs

    def _classify_digit(self, crop):
        """Match a binarized digit crop against the templates."""
        crop = cv2.resize(crop, DIGIT_SIZE)
        best_digit, best_score = None, -1.0
        for digit, template in self.templates.items():
            score = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)[0][0]
            if score > best_score:
                best_digit, best_score = digit, float(score)
        return best_digit, best_score

    def read(self, frame):
        """
        Read the remaining-cards counts from a full game frame.

        Args:
            frame: BGR image of the whole game screen

        Returns:
            Dict mapping each Rank to its remaining count, or None when
            the bar is not visible (anything other than exactly one
            digit per column is treated as "not readable").
        """
        height, width = frame.shape[:2]
        bar = frame[int(height * BAR_REGION['top']):int(height * BAR_REGION['bottom']),
                    int(width * BAR_REGION['left']):int(width * BAR_REGION['right'])]

        mask = self._digit_mask(bar)
        blobs = self._find_digit_blobs(mask)

        if len(blobs) != len(RANK_ORDER):
            return None

        counts = {}
        for rank, (x, y, w, h) in zip(RANK_ORDER, blobs):
            digit, _ = self._classify_digit(mask[y:y + h, x:x + w])
            counts[rank] = digit
        return counts


_reader = None


def read_status_bar(frame):
    """Module-level convenience wrapper with a cached reader."""
    global _reader
    if _reader is None:
        _reader = StatusBarReader()
    return _reader.read(frame)
