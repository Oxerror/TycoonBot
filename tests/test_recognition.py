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


class TestDimMask:
    """The dim mask keeps the glyphs of greyed-out cards: on the
    player's turn the game dims invalid-to-play cards to a uniform
    ~56%, landing their white glyphs at neutral gray ~142. It is the
    hand's second matching pass — bright glyphs belong to the white
    mask pass, which stays untouched."""

    def test_keeps_dim_glyphs_only(self, recognizer):
        image = np.full((10, 10, 3), 50, dtype=np.uint8)
        image[2:4, 2:4] = 250          # bright glyph: white pass' job
        image[6:8, 6:8] = 142          # greyed-out glyph
        masked = recognizer.apply_dim_mask(image)
        assert (masked[2:4, 2:4] == 0).all()
        assert (masked[6:8, 6:8] == 255).all()
        assert (masked[0:2, 0:2] == 0).all()

    def test_saturated_dim_pixels_stay_out(self, recognizer):
        """A dimmed red card background is as bright as a dimmed glyph
        but keeps its saturation — it must not enter the mask."""
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[:, :] = (40, 40, 150)    # dimmed red, BGR
        masked = recognizer.apply_dim_mask(image)
        assert (masked == 0).all()

    def test_white_mask_alone_erases_dim_glyphs(self, recognizer):
        """Documents why the hand needs the second pass — and why the
        field must keep the plain white mask (dim = an old trick)."""
        image = np.full((4, 4, 3), 142, dtype=np.uint8)
        assert (recognizer.apply_white_mask(image) == 0).all()


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
    'TestImage5.png': [
        (Rank.SEVEN, Suit.CLUBS),
        (Rank.SEVEN, Suit.HEARTS),
        (Rank.EIGHT, Suit.CLUBS),
        (Rank.EIGHT, Suit.SPADES),
        (Rank.TEN, Suit.DIAMONDS),
        (Rank.JACK, Suit.DIAMONDS),
        (Rank.QUEEN, Suit.DIAMONDS),
        (Rank.QUEEN, Suit.SPADES),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.KING, Suit.HEARTS),
        (Rank.KING, Suit.SPADES),
        (Rank.TWO, Suit.CLUBS),
        (Rank.TWO, Suit.DIAMONDS),
    ],
    'TestImage6.png': [
        (Rank.EIGHT, Suit.CLUBS),
        (Rank.EIGHT, Suit.SPADES),
        (Rank.TEN, Suit.DIAMONDS),
        (Rank.JACK, Suit.DIAMONDS),
        (Rank.QUEEN, Suit.DIAMONDS),
        (Rank.QUEEN, Suit.SPADES),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.KING, Suit.HEARTS),
        (Rank.KING, Suit.SPADES),
        (Rank.TWO, Suit.CLUBS),
        (Rank.TWO, Suit.DIAMONDS),
    ],
    # Player already went out: the hand area is empty.
    'TestImage7.png': [],
    # Small hand where the Wonder and Joker emblems are fully visible.
    'TestImage8.png': [
        (Rank.WONDER, None),
        (Rank.SEVEN, Suit.DIAMONDS),
        (Rank.EIGHT, Suit.HEARTS),
        (Rank.QUEEN, Suit.CLUBS),
        (Rank.KING, Suit.CLUBS),
        (Rank.JOKER, None),
    ],
    # Revolution active (see the indicator test); small hand with both
    # specials and the 3 of Spades fully readable.
    'TestImage10.png': [
        (Rank.WONDER, None),
        (Rank.THREE, Suit.SPADES),
        (Rank.EIGHT, Suit.SPADES),
        (Rank.JOKER, None),
    ],
    # Round start, 14 cards: the Wonder and Joker at the fan edges are
    # clipped beyond recognition (the start-hand validation test below
    # recovers them from the status bar).
    'TestImage9.png': [
        (Rank.FIVE, Suit.DIAMONDS),
        (Rank.FIVE, Suit.HEARTS),
        (Rank.SIX, Suit.CLUBS),
        (Rank.SEVEN, Suit.DIAMONDS),
        (Rank.EIGHT, Suit.HEARTS),
        (Rank.JACK, Suit.CLUBS),
        (Rank.QUEEN, Suit.CLUBS),
        (Rank.KING, Suit.CLUBS),
        (Rank.ACE, Suit.CLUBS),
        (Rank.ACE, Suit.HEARTS),
        (Rank.TWO, Suit.CLUBS),
        (Rank.TWO, Suit.DIAMONDS),
    ],
}


@pytest.mark.slow
@pytest.mark.parametrize('image_name', sorted(EXPECTED_HANDS))
def test_reads_full_hand_from_screenshot(image_name):
    """End to end: screenshot -> detections -> Cards.

    Only the recognizable cards are expected: in dense fans the
    outermost cards (Wonder/Joker) are clipped beyond their emblems and
    are absent from the expectations of those fixtures.
    """
    image = cv2.imread(str(IMAGE_DIR / image_name))
    assert image is not None, f'missing test image {image_name}'

    height = image.shape[0]
    hand_region = image[height * 3 // 4:height]

    cards = read_hand(hand_region)

    assert [(c.rank, c.suit) for c in cards] == EXPECTED_HANDS[image_name]


# The game dims earlier plays, so only the current (bright) trick is
# expected. In TestImage the 5-5 pair is still bright alongside the
# just-played Joker; the 5 of clubs' rank glyph is too eroded to read.
EXPECTED_TRICKS = {
    'TestImage.png': [(Rank.FIVE, Suit.HEARTS), (Rank.JOKER, None)],
    'TestImage2.png': [(Rank.TWO, Suit.HEARTS)],
    'TestImage5.png': [(Rank.FOUR, Suit.DIAMONDS), (Rank.FOUR, Suit.SPADES)],
    # A double-joker play: the rear joker's emblem is covered by the
    # front card, so only one Joker is detectable.
    'TestImage6.png': [(Rank.JOKER, None)],
    'TestImage7.png': [(Rank.EIGHT, Suit.HEARTS), (Rank.EIGHT, Suit.SPADES)],
    'TestImage8.png': [(Rank.EIGHT, Suit.SPADES)],
    # Round start: nothing on the table yet.
    'TestImage9.png': [],
    'TestImage10.png': [(Rank.FOUR, Suit.HEARTS), (Rank.FOUR, Suit.SPADES)],
    # A real Wonder played onto the table.
    'TestImage11.png': [(Rank.WONDER, None)],
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


def test_event_banner_detection():
    """Event banners cover the field with white shapes that fool the
    card templates (the Done star reads as a Wonder), so banner frames
    must be recognized and their field reading skipped."""
    from ImageRecognition import banner_visible

    banner = cv2.imread(str(IMAGE_DIR / 'TestImage12.png'))
    height, width = banner.shape[:2]
    field = banner[int(height * 0.4):int(height * 0.8),
                   int(width * 0.333):int(width * 0.667)]
    assert banner_visible(field) is True

    for name in ['TestImage.png', 'TestImage10.png', 'TestImage11.png']:
        image = cv2.imread(str(IMAGE_DIR / name))
        height, width = image.shape[:2]
        field = image[int(height * 0.4):int(height * 0.8),
                      int(width * 0.333):int(width * 0.667)]
        assert banner_visible(field) is False, name


def test_revolution_indicator():
    """The persistent Flip Strength badge marks an active revolution."""
    from ImageRecognition import read_revolution_indicator

    active = cv2.imread(str(IMAGE_DIR / 'TestImage10.png'))
    assert read_revolution_indicator(active) is True

    for name in ['TestImage.png', 'TestImage7.png', 'TestImage9.png']:
        image = cv2.imread(str(IMAGE_DIR / name))
        assert read_revolution_indicator(image) is False, name


def read_capture(name):
    """A frame from the (gitignored) recorded session, or skip."""
    path = IMAGE_DIR / 'captures' / name
    if not path.exists():
        pytest.skip("no recorded captures on this machine")
    image = cv2.imread(str(path))
    assert image is not None
    return image


# The hand crop used by the live loop (config.json hand_region): it
# reaches above the resting fan so that selected cards, which lift
# upward out of the old 0.75 crop, keep their glyphs in view.
HAND_CROP_TOP = 0.72


@pytest.mark.slow
def test_greyed_out_cards_are_recognized():
    """On the player's turn the game dims every card that cannot join
    a valid play; those cards are still in the hand. This frame holds
    14 cards, six of them greyed out (a triple of 4s is on the table),
    with only the fan-edge Wonder and Joker clipped beyond reading."""
    image = read_capture('20260808_215057_interval.png')
    height = image.shape[0]
    cards = read_hand(image[int(height * HAND_CROP_TOP):])

    assert [(c.rank, c.suit) for c in cards] == [
        (Rank.FIVE, Suit.DIAMONDS),
        (Rank.FIVE, Suit.HEARTS),
        (Rank.SIX, Suit.CLUBS),        # greyed out from here...
        (Rank.SEVEN, Suit.DIAMONDS),
        (Rank.EIGHT, Suit.HEARTS),
        (Rank.JACK, Suit.CLUBS),
        (Rank.QUEEN, Suit.CLUBS),
        (Rank.KING, Suit.CLUBS),       # ...to here
        (Rank.ACE, Suit.CLUBS),
        (Rank.ACE, Suit.HEARTS),
        (Rank.TWO, Suit.CLUBS),
        (Rank.TWO, Suit.DIAMONDS),
    ]


@pytest.mark.slow
def test_lifted_selected_cards_stay_in_the_hand_reading():
    """Selecting cards lifts them ~55px above the fan; the crop must
    still contain their glyphs. Here two 2s are lifted: the 2 of
    diamonds reads fine, while the 2 of clubs' glyph is partially
    covered by the neighboring lifted card — occlusion recognition
    cannot beat, so the tracker carries that one via the counter."""
    image = read_capture('20260808_215508_unknown-digit.png')
    height = image.shape[0]
    cards = read_hand(image[int(height * HAND_CROP_TOP):])

    read = [(c.rank, c.suit) for c in cards]
    assert (Rank.TWO, Suit.DIAMONDS) in read, "lifted card fell out of the crop"
    for expected in [(Rank.WONDER, None), (Rank.QUEEN, Suit.DIAMONDS),
                     (Rank.QUEEN, Suit.SPADES)]:
        assert expected in read


@pytest.mark.slow
def test_start_hand_validation_recovers_clipped_cards():
    """At round start deck - bar = own hand, so the status bar proves
    which cards the fan clipped away: TestImage9's hand reading lacks
    exactly the edge Wonder and Joker."""
    from GameLogic.GameState import validate_start_hand
    from StatusBarReader import read_status_bar

    image = cv2.imread(str(IMAGE_DIR / 'TestImage9.png'))
    assert image is not None

    height = image.shape[0]
    cards = read_hand(image[height * 3 // 4:height])
    bar_counts = read_status_bar(image)

    missing, extra = validate_start_hand(cards, bar_counts)

    assert missing == {Rank.WONDER: 1, Rank.JOKER: 1}
    assert extra == {}
