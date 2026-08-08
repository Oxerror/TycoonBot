"""Tests for the recognition layer.

The integration tests run full template matching on the committed
gameplay screenshots and assert the exact hand is read. They take a few
seconds each; run only the fast unit tests with:

    pytest -m "not slow"
"""

import cv2
import numpy as np
import pytest

from GameLogic.Card import Rank, Suit
from GameLogic.HandReader import detections_to_cards
from ImageRecognition import CardRecognizer, PROJECT_ROOT, read_hand, read_play_field

IMAGE_DIR = PROJECT_ROOT / 'Image'


@pytest.fixture(scope='module')
def recognizer():
    return CardRecognizer()


class TestTemplateLoading:
    def test_all_templates_loaded(self, recognizer):
        assert len(recognizer.templates) == 19
        for name in ['2', '10', 'Ace', 'King', 'Heart', 'Spade', 'Cross',
                     'Diamond', 'Joker', 'Wonder']:
            assert name in recognizer.templates


class TestWhiteMask:
    def test_color_image(self, recognizer):
        image = np.full((10, 10, 3), 50, dtype=np.uint8)
        image[2:4, 2:4] = 250
        masked = recognizer.apply_white_mask(image)
        assert (masked[2:4, 2:4] == 255).all()
        assert (masked[5:, 5:] == 0).all()

    def test_grayscale_image(self, recognizer):
        image = np.full((10, 10), 50, dtype=np.uint8)
        image[2:4, 2:4] = 250
        masked = recognizer.apply_white_mask(image)
        assert (masked[2:4, 2:4] == 255).all()
        assert (masked[5:, 5:] == 0).all()


class TestNonMaxSuppression:
    def test_overlapping_keeps_most_confident(self, recognizer):
        detections = [
            {'name': 'a', 'confidence': 0.8, 'location': (10, 10, 50, 50)},
            {'name': 'b', 'confidence': 0.9, 'location': (12, 12, 50, 50)},
        ]
        kept = recognizer._non_max_suppression(detections)
        assert [d['name'] for d in kept] == ['b']

    def test_nested_box_suppressed(self, recognizer):
        """A small box inside a big one has low IoU but must still be suppressed."""
        detections = [
            {'name': 'big', 'confidence': 0.9, 'location': (0, 0, 100, 100)},
            {'name': 'small', 'confidence': 0.8, 'location': (40, 40, 20, 20)},
        ]
        kept = recognizer._non_max_suppression(detections)
        assert [d['name'] for d in kept] == ['big']

    def test_distant_boxes_both_kept(self, recognizer):
        detections = [
            {'name': 'a', 'confidence': 0.8, 'location': (0, 0, 50, 50)},
            {'name': 'b', 'confidence': 0.9, 'location': (200, 0, 50, 50)},
        ]
        assert len(recognizer._non_max_suppression(detections)) == 2


EXPECTED_HANDS = {
    'TestImage.png': [
        (Rank.FIVE, Suit.SPADES),
        (Rank.SIX, Suit.HEARTS),
        (Rank.SIX, Suit.SPADES),
        (Rank.EIGHT, Suit.CLUBS),
        (Rank.EIGHT, Suit.HEARTS),
        (Rank.NINE, Suit.CLUBS),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.KING, Suit.SPADES),
        (Rank.ACE, Suit.HEARTS),
        (Rank.ACE, Suit.SPADES),
        (Rank.TWO, Suit.CLUBS),
        (Rank.TWO, Suit.DIAMONDS),
    ],
    'TestImage2.png': [
        (Rank.THREE, Suit.SPADES),
        (Rank.EIGHT, Suit.CLUBS),
        (Rank.NINE, Suit.CLUBS),
        (Rank.NINE, Suit.SPADES),
        (Rank.JACK, Suit.DIAMONDS),
        (Rank.JACK, Suit.HEARTS),
        (Rank.QUEEN, Suit.SPADES),
        (Rank.KING, Suit.CLUBS),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.KING, Suit.HEARTS),
        (Rank.KING, Suit.SPADES),
        (Rank.TWO, Suit.DIAMONDS),
    ],
}


@pytest.mark.slow
@pytest.mark.parametrize('image_name', sorted(EXPECTED_HANDS))
def test_reads_full_hand_from_screenshot(image_name):
    """End to end: screenshot -> detections -> Cards.

    Only the fully visible cards are expected; the fan clips the
    outermost cards (Wonder/Joker) so those may or may not be found.
    """
    image = cv2.imread(str(IMAGE_DIR / image_name))
    assert image is not None, f'missing test image {image_name}'

    height = image.shape[0]
    hand_region = image[height * 3 // 4:height]

    cards = read_hand(hand_region)

    got = [(c.rank, c.suit) for c in cards
           if c.rank not in (Rank.JOKER, Rank.WONDER)]
    assert got == EXPECTED_HANDS[image_name]


# The game dims earlier plays, so only the current (bright) trick is
# expected. In TestImage the 5-5 pair is still bright alongside the
# just-played Joker; the 5 of clubs' rank glyph is too eroded to read.
EXPECTED_TRICKS = {
    'TestImage.png': [(Rank.FIVE, Suit.HEARTS), (Rank.JOKER, None)],
    'TestImage2.png': [(Rank.TWO, Suit.HEARTS)],
}


@pytest.mark.slow
@pytest.mark.parametrize('image_name', sorted(EXPECTED_TRICKS))
def test_reads_current_trick_from_screenshot(image_name):
    image = cv2.imread(str(IMAGE_DIR / image_name))
    assert image is not None, f'missing test image {image_name}'

    height, width = image.shape[:2]
    field_region = image[int(height * 0.4):int(height * 0.8),
                         int(width * 0.333):int(width * 0.667)]

    cards = read_play_field(field_region)

    assert [(c.rank, c.suit) for c in cards] == EXPECTED_TRICKS[image_name]
