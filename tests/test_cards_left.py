import cv2
import numpy as np
import pytest

from CardsLeftReader import CardsLeftReader, read_cards_left
from ImageRecognition import PROJECT_ROOT

IMAGE_DIR = PROJECT_ROOT / 'Image'

EXPECTED = {
    'TestImage.png': {'left': 14, 'middle': 11, 'right': 8, 'player': 14},
    'TestImage2.png': {'left': 12, 'middle': 10, 'right': 13, 'player': 13},
}


class TestDigitTemplates:
    def test_known_digits_loaded(self):
        reader = CardsLeftReader()
        assert sorted(reader.templates) == [0, 1, 2, 3, 4, 8]

    def test_templates_classify_as_themselves(self):
        reader = CardsLeftReader()
        for digit, template in reader.templates.items():
            assert reader._classify_digit(template) == digit


@pytest.mark.parametrize('image_name', sorted(EXPECTED))
def test_reads_all_counters(image_name):
    image = cv2.imread(str(IMAGE_DIR / image_name))
    assert image is not None, f'missing test image {image_name}'
    assert read_cards_left(image) == EXPECTED[image_name]


def test_non_game_frame_reads_none():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert read_cards_left(frame) == {
        'left': None, 'middle': None, 'right': None, 'player': None,
    }
