import cv2
import numpy as np
import pytest

from CardsLeftReader import (CardsLeftReader, read_cards_left,
                             read_cards_left_detailed)
from ImageRecognition import PROJECT_ROOT

IMAGE_DIR = PROJECT_ROOT / 'Image'

EXPECTED = {
    'TestImage.png': {'left': 14, 'middle': 11, 'right': 8, 'player': 14},
    'TestImage2.png': {'left': 12, 'middle': 10, 'right': 13, 'player': 13},
    'TestImage5.png': {'left': 14, 'middle': 12, 'right': 12, 'player': 14},
    'TestImage6.png': {'left': 12, 'middle': 10, 'right': 10, 'player': 12},
    # Left and right exercise the 7 and 6 templates from the captures;
    # 6 was initially misread as 8 until its template existed, which the
    # bar cross-check exposed (bar total 17 vs a claimed 7+4+8).
    'TestImage7.png': {'left': 7, 'middle': 4, 'right': 6, 'player': 0},
    'TestImage8.png': {'left': 8, 'middle': 6, 'right': 10, 'player': 6},
    # Round start: 56 dealt cards, 14 each.
    'TestImage9.png': {'left': 14, 'middle': 14, 'right': 14, 'player': 14},
    # The left bubble is active (enlarged): its title letter must not
    # spoil the digit segmentation.
    'TestImage10.png': {'left': 10, 'middle': 10, 'right': 10, 'player': 4},
}


class TestDigitTemplates:
    def test_known_digits_loaded(self):
        reader = CardsLeftReader()
        assert sorted(reader.templates) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_templates_classify_as_themselves(self):
        reader = CardsLeftReader()
        for digit, template in reader.templates.items():
            assert reader._classify_digit(template) == digit


# Whose bubble carries the red your-turn marker in each fixture.
EXPECTED_ACTIVE = {
    'TestImage.png': 'right',
    'TestImage2.png': 'right',
    'TestImage5.png': 'right',
    'TestImage6.png': 'right',
    'TestImage7.png': 'left',
    'TestImage8.png': 'middle',
    'TestImage9.png': 'middle',
    'TestImage10.png': 'left',
}


@pytest.mark.parametrize('image_name', sorted(EXPECTED))
def test_reads_all_counters(image_name):
    image = cv2.imread(str(IMAGE_DIR / image_name))
    assert image is not None, f'missing test image {image_name}'
    assert read_cards_left(image) == EXPECTED[image_name]


@pytest.mark.parametrize('image_name', sorted(EXPECTED_ACTIVE))
def test_reads_active_player(image_name):
    image = cv2.imread(str(IMAGE_DIR / image_name))
    assert image is not None, f'missing test image {image_name}'
    _, _, active, _ = read_cards_left_detailed(image)
    assert active == EXPECTED_ACTIVE[image_name]


# No fixture shows a pass marker: the yellow reading is exercised by
# TestImage13 below; these frames were all taken outside a pass.
@pytest.mark.parametrize('image_name', sorted(EXPECTED))
def test_no_pass_marker_on_regular_fixtures(image_name):
    image = cv2.imread(str(IMAGE_DIR / image_name))
    assert image is not None, f'missing test image {image_name}'
    _, _, _, passed = read_cards_left_detailed(image)
    assert passed == []


def test_reads_pass_markers_and_their_counts():
    """TestImage13: the middle opponent has passed — yellow, enlarged
    bubble text the white-mask reader could not even count. The own
    counter reads 0: the player already finished this round."""
    image = cv2.imread(str(IMAGE_DIR / 'TestImage13.png'))
    assert image is not None, 'missing test image TestImage13.png'
    counts, unknown, active, passed = read_cards_left_detailed(image)
    assert passed == ['middle']
    assert counts == {'left': 2, 'middle': 1, 'right': 2, 'player': 0}
    assert active == 'left'


def test_no_active_player_on_blank_frame():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    _, _, active, passed = read_cards_left_detailed(frame)
    assert active is None
    assert passed == []


class TestPlayerTurnButtons:
    """The player's own red marker fails on two-digit counts (the
    digits sit on the badge's black wedge), so the Pass/Hint button
    row is the primary your-turn signal. Uses recorded captures —
    skips on a fresh clone."""

    @staticmethod
    def read_capture(name):
        path = IMAGE_DIR / 'captures' / name
        if not path.exists():
            pytest.skip("no recorded captures on this machine")
        image = cv2.imread(str(path))
        assert image is not None
        return image

    def test_two_digit_own_turn_is_detected(self):
        image = self.read_capture('20260808_215057_interval.png')
        counts, _, active, _ = read_cards_left_detailed(image)
        assert counts['player'] == 14
        assert active == 'player'

    def test_turn_start_before_the_buttons_fade_in(self):
        # The button row is not drawn yet on this frame; the red badge
        # marker (single-digit count) must still catch the turn.
        image = self.read_capture('20260808_215448_unknown-digit.png')
        _, _, active, _ = read_cards_left_detailed(image)
        assert active == 'player'


def test_non_game_frame_reads_none():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert read_cards_left(frame) == {
        'left': None, 'middle': None, 'right': None, 'player': None,
    }
