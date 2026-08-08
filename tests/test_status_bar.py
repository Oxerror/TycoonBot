import cv2
import numpy as np
import pytest

from GameLogic.Card import Rank
from StatusBarReader import RANK_ORDER, StatusBarReader, read_status_bar
from ImageRecognition import PROJECT_ROOT

IMAGE_DIR = PROJECT_ROOT / 'Image'

EXPECTED_COUNTS = {
    'TestImage.png': [1, 1, 4, 1, 2, 4, 2, 3, 1, 4, 4, 2, 2, 2, 0],
    'TestImage2.png': [1, 0, 4, 4, 4, 4, 1, 2, 3, 2, 3, 0, 3, 2, 2],
    'TestImage5.png': [1, 2, 2, 4, 4, 2, 2, 4, 3, 3, 2, 1, 4, 2, 2],
    'TestImage6.png': [1, 2, 2, 4, 4, 2, 2, 2, 3, 3, 2, 1, 2, 2, 0],
    'TestImage7.png': [0, 1, 1, 2, 2, 3, 1, 0, 2, 3, 0, 2, 0, 0, 0],
    'TestImage8.png': [0, 1, 0, 0, 2, 1, 2, 2, 4, 3, 3, 3, 1, 1, 1],
    'TestImage9.png': [1, 4, 4, 2, 3, 3, 3, 4, 4, 3, 3, 3, 2, 2, 1],
}


class TestRankOrder:
    def test_fifteen_columns(self):
        assert len(RANK_ORDER) == 15

    def test_covers_every_rank(self):
        assert set(RANK_ORDER) == set(Rank)


class TestDigitClassification:
    def test_templates_classify_as_themselves(self):
        reader = StatusBarReader()
        assert sorted(reader.templates) == [0, 1, 2, 3, 4]
        for digit, template in reader.templates.items():
            got, score = reader._classify_digit(template)
            assert got == digit
            assert score > 0.99


@pytest.mark.parametrize('image_name', sorted(EXPECTED_COUNTS))
def test_reads_counts_from_screenshot(image_name):
    image = cv2.imread(str(IMAGE_DIR / image_name))
    assert image is not None, f'missing test image {image_name}'

    counts = read_status_bar(image)

    assert counts is not None
    assert [counts[rank] for rank in RANK_ORDER] == EXPECTED_COUNTS[image_name]


def test_returns_none_without_bar():
    """A frame that is not the game must not produce fantasy counts."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert read_status_bar(frame) is None


def test_returns_none_on_unrelated_image():
    image = cv2.imread(str(IMAGE_DIR / 'TestImage4.png'))
    assert image is not None
    assert read_status_bar(image) is None
