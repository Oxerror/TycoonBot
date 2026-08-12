"""Read the "Cards Left" counters shown next to each player.

Each player's speech bubble shows how many cards they hold. The number
is drawn in large white digits (taller than the surrounding "Cards
Left" lettering), so digits are segmented by a white mask plus a
height filter and classified against digit templates extracted from
the reference screenshots.

The template set covers all ten digits. A blob that matches no
template makes the affected counter read as None instead of guessing.

The bubble also announces whose turn it is: the active player's bubble
turns bright red while the others stay black. The area around the
digits is sampled (ignoring the white text) to read that marker.

The red marker is unreliable for the player's own badge when the count
has two digits (the digits sit on the badge's black wedge, diluting
the sampled redness below the threshold — measured live 2026-08-12
with 12 cards: 90 vs the 100 cutoff). The Pass/Hint button row that
the game draws while waiting for the player's move is therefore the
primary your-turn signal; the redness still catches the turn's first
frames, before the row has faded in.

A player who passed keeps their bubble text in yellow until the trick
ends (seen throughout the recorded sessions, e.g. two opponents yellow
at once mid-trick), so the same digit segmentation run on a yellow
mask reads both the count and the pass state of players the white
mask cannot see.
"""

import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DIGIT_TEMPLATE_DIR = PROJECT_ROOT / 'Image' / 'templates' / 'bubble_digits'
TURN_BUTTON_DIR = PROJECT_ROOT / 'Image' / 'templates' / 'turn_buttons'

# Bubble regions as fractions of the full frame (measured at 1080p).
PLAYER_REGIONS = {
    'left': (0.052, 0.167, 0.167, 0.306),
    'middle': (0.323, 0.469, 0.111, 0.241),
    'right': (0.755, 0.911, 0.157, 0.306),
    'player': (0.031, 0.167, 0.704, 0.833),
}

# Count digits are noticeably taller than the "Cards Left" lettering.
# Measured at 1080p and scaled with the frame height. The yellow pass
# text is drawn larger than the white version (digits 46-50px).
DIGIT_HEIGHT = (36, 42)
YELLOW_DIGIT_HEIGHT = (44, 54)
DIGIT_MAX_WIDTH = 45
REFERENCE_FRAME_HEIGHT = 1080

DIGIT_SIZE = (32, 38)

# Below this match score a blob is not any known digit.
MIN_DIGIT_SCORE = 0.6

# Mean (red - max(green, blue)) of the bubble area around the digits;
# an active bubble measures ~250, an inactive one ~0.
ACTIVE_BUBBLE_REDNESS = 100

# Band of the frame holding the Pass / timer / Hint / Select row on
# the player's turn, as (fy1, fy2, fx1, fx2). Measured over the
# recorded captures: my-turn frames match the banner templates at
# >= 0.96, every other frame at <= 0.4.
TURN_BUTTON_BAND = (0.62, 0.76, 0.26, 0.62)
TURN_BUTTON_SCORE = 0.8


class CardsLeftReader:
    def __init__(self, template_dir=DIGIT_TEMPLATE_DIR,
                 button_dir=TURN_BUTTON_DIR):
        self.templates = {}
        for template_file in Path(template_dir).glob('*.png'):
            digit = int(template_file.stem)
            image = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            self.templates[digit] = cv2.resize(image, DIGIT_SIZE)
        if not self.templates:
            raise FileNotFoundError(f"No digit templates in {template_dir}")
        self.turn_templates = [
            image for image in (cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                                for p in Path(button_dir).glob('*.png'))
            if image is not None]
        if not self.turn_templates:
            raise FileNotFoundError(f"No turn-button templates in {button_dir}")

    @staticmethod
    def _white_mask(crop):
        b, g, r = cv2.split(crop.astype(np.int16))
        return ((r > 190) & (g > 190) & (b > 190)).astype(np.uint8) * 255

    @staticmethod
    def _yellow_mask(crop):
        """The pass marker: bubble text turns yellow (measured ~BGR
        96,228,232) until the trick ends."""
        b, g, r = cv2.split(crop.astype(np.int16))
        return ((r > 180) & (g > 150) & (b < 140)
                & (g - b > 60)).astype(np.uint8) * 255

    @staticmethod
    def _find_digit_blobs(mask, frame_height, height_range=DIGIT_HEIGHT):
        scale = frame_height / REFERENCE_FRAME_HEIGHT
        min_h, max_h = height_range[0] * scale, height_range[1] * scale
        max_w = DIGIT_MAX_WIDTH * scale

        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        blobs = []
        for i in range(1, count):
            x, y, w, h, _ = stats[i]
            if min_h <= h <= max_h and 10 * scale <= w <= max_w:
                blobs.append((x, y, w, h))

        if len(blobs) > 2:
            # An active (enlarged) bubble can push a title letter into
            # the digit size range. The count digits share a baseline
            # and sit rightmost, so keep only that cluster.
            clusters = []
            for blob in sorted(blobs, key=lambda b: b[1]):
                for cluster in clusters:
                    if abs(cluster[0][1] - blob[1]) <= 12 * scale:
                        cluster.append(blob)
                        break
                else:
                    clusters.append([blob])
            blobs = max(clusters, key=lambda c: max(b[0] for b in c))

        blobs.sort()
        return blobs

    def _classify_digit(self, crop):
        crop = cv2.resize(crop, DIGIT_SIZE)
        best_digit, best_score = None, -1.0
        for digit, template in self.templates.items():
            score = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)[0][0]
            if score > best_score:
                best_digit, best_score = digit, float(score)
        if best_score < MIN_DIGIT_SCORE:
            return None
        return best_digit

    @staticmethod
    def _bubble_redness(crop, mask, blobs):
        """Mean red dominance of the bubble area around the digits,
        ignoring the white text pixels themselves."""
        pad = 25
        x1 = max(0, min(b[0] for b in blobs) - pad)
        y1 = max(0, min(b[1] for b in blobs) - pad)
        x2 = min(crop.shape[1], max(b[0] + b[2] for b in blobs) + pad)
        y2 = min(crop.shape[0], max(b[1] + b[3] for b in blobs) + pad)

        patch = crop[y1:y2, x1:x2].astype(np.int16)
        background = patch[mask[y1:y2, x1:x2] == 0]
        if background.size == 0:
            return 0.0
        blue, green, red = (background[:, 0].mean(),
                            background[:, 1].mean(),
                            background[:, 2].mean())
        return red - max(green, blue)

    def _player_turn_buttons(self, frame):
        """True while the Pass/Hint button row is on screen — the game
        draws it only when waiting for the player's move."""
        height, width = frame.shape[:2]
        fy1, fy2, fx1, fx2 = TURN_BUTTON_BAND
        band = frame[int(height * fy1):int(height * fy2),
                     int(width * fx1):int(width * fx2)]
        band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        if height != REFERENCE_FRAME_HEIGHT:
            scale = REFERENCE_FRAME_HEIGHT / height
            band = cv2.resize(band, None, fx=scale, fy=scale)
        if float(band.std()) < 10:
            # A near-flat band cannot hold the banners, and normalized
            # matching is numerically unstable on zero-variance input.
            return False
        return any(cv2.matchTemplate(band, template,
                                     cv2.TM_CCOEFF_NORMED).max()
                   >= TURN_BUTTON_SCORE for template in self.turn_templates)

    def read_detailed(self, frame):
        """
        Read every player's card counter from a full game frame.

        Returns:
            Tuple (counts, unknown, active, passed): counts maps
            {'left', 'middle', 'right', 'player'} to int or None when
            that counter is not readable; unknown lists the players
            whose counter showed a digit-sized blob matching no known
            digit template; active names the player whose bubble
            carries the red your-turn marker ('player' also when the
            Pass/Hint button row is on screen), or None when neither
            marker shows; passed lists the players whose bubble text is
            yellow — the pass marker shown until the trick ends.
        """
        height, width = frame.shape[:2]
        counts = {}
        unknown = []
        active = None
        passed = []

        for player, (fx1, fx2, fy1, fy2) in PLAYER_REGIONS.items():
            crop = frame[int(height * fy1):int(height * fy2),
                         int(width * fx1):int(width * fx2)]
            mask = self._white_mask(crop)
            blobs = self._find_digit_blobs(mask, height)
            has_passed = False

            if not blobs or len(blobs) > 2:
                yellow = self._yellow_mask(crop)
                yellow_blobs = self._find_digit_blobs(yellow, height,
                                                      YELLOW_DIGIT_HEIGHT)
                if yellow_blobs and len(yellow_blobs) <= 2:
                    mask, blobs = yellow, yellow_blobs
                    has_passed = True
                    passed.append(player)

            if not blobs or len(blobs) > 2:
                counts[player] = None
                continue

            if (not has_passed
                    and self._bubble_redness(crop, mask, blobs)
                    > ACTIVE_BUBBLE_REDNESS):
                active = player

            digits = [self._classify_digit(mask[y:y + h, x:x + w])
                      for x, y, w, h in blobs]
            if None in digits:
                counts[player] = None
                unknown.append(player)
            else:
                counts[player] = int(''.join(str(d) for d in digits))

        if active is None and self._player_turn_buttons(frame):
            active = 'player'

        return counts, unknown, active, passed

    def read(self, frame):
        """Like read_detailed, but returns only the counts dict."""
        return self.read_detailed(frame)[0]


_reader = None


def read_cards_left(frame):
    """Module-level convenience wrapper with a cached reader."""
    global _reader
    if _reader is None:
        _reader = CardsLeftReader()
    return _reader.read(frame)


def read_cards_left_detailed(frame):
    """Like read_cards_left, but also returns the unknown-digit players."""
    global _reader
    if _reader is None:
        _reader = CardsLeftReader()
    return _reader.read_detailed(frame)
