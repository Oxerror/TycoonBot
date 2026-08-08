import cv2
import numpy as np

from CaptureData import frames_differ
from ScreenCapture import applyRedactions, cropRegion


class TestRedaction:
    def test_region_is_blacked_out(self):
        frame = np.full((100, 200, 3), 255, dtype=np.uint8)
        applyRedactions(frame, [{'top': 0.9, 'bottom': 1.0, 'left': 0.8, 'right': 1.0}])
        assert (frame[90:, 160:] == 0).all()

    def test_rest_of_frame_untouched(self):
        frame = np.full((100, 200, 3), 255, dtype=np.uint8)
        applyRedactions(frame, [{'top': 0.9, 'bottom': 1.0, 'left': 0.8, 'right': 1.0}])
        assert (frame[:90, :] == 255).all()
        assert (frame[:, :160] == 255).all()

    def test_multiple_regions(self):
        frame = np.full((100, 200, 3), 255, dtype=np.uint8)
        applyRedactions(frame, [
            {'top': 0.0, 'bottom': 0.1, 'left': 0.0, 'right': 0.1},
            {'top': 0.9, 'bottom': 1.0, 'left': 0.9, 'right': 1.0},
        ])
        assert (frame[:10, :20] == 0).all()
        assert (frame[90:, 180:] == 0).all()

    def test_no_regions_is_a_noop(self):
        frame = np.full((100, 200, 3), 255, dtype=np.uint8)
        applyRedactions(frame, [])
        assert (frame == 255).all()

    def test_redaction_happens_in_place(self):
        """The caller's frame itself must be redacted, so an unredacted
        copy never exists."""
        frame = np.full((100, 200, 3), 255, dtype=np.uint8)
        result = applyRedactions(frame, [{'top': 0.0, 'bottom': 1.0, 'left': 0.0, 'right': 1.0}])
        assert result is frame


class TestCropRegion:
    def test_fractional_crop(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = cropRegion(frame, {'top': 0.5, 'bottom': 1.0, 'left': 0.25, 'right': 0.75})
        assert crop.shape == (50, 100, 3)


class TestFramesDiffer:
    def test_identical_frames_do_not_differ(self):
        frame = np.random.default_rng(1).integers(0, 255, (90, 160, 3), dtype=np.uint8)
        assert not frames_differ(frame, frame.copy())

    def test_none_always_differs(self):
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        assert frames_differ(frame, None)

    def test_changed_frames_differ(self):
        frame_a = np.zeros((90, 160, 3), dtype=np.uint8)
        frame_b = frame_a.copy()
        frame_b[20:60, 40:120] = 255
        assert frames_differ(frame_a, frame_b)

    def test_real_screenshots_differ(self):
        from ImageRecognition import PROJECT_ROOT
        a = cv2.imread(str(PROJECT_ROOT / 'Image' / 'TestImage.png'))
        b = cv2.imread(str(PROJECT_ROOT / 'Image' / 'TestImage2.png'))
        assert frames_differ(a, b)
        assert not frames_differ(a, a.copy())


class TestUnknownDigitDetection:
    def test_unreadable_digit_blob_is_flagged(self):
        from CardsLeftReader import CardsLeftReader
        reader = CardsLeftReader()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Draw a digit-sized X inside the left opponent's bubble region;
        # it segments like a digit but matches no digit template.
        cv2.line(frame, (200, 242), (228, 276), (255, 255, 255), 5)
        cv2.line(frame, (228, 242), (200, 276), (255, 255, 255), 5)
        counts, unknown = reader.read_detailed(frame)
        assert counts['left'] is None
        assert unknown == ['left']

    def test_clean_screenshots_have_no_unknown_digits(self):
        from CardsLeftReader import read_cards_left_detailed
        from ImageRecognition import PROJECT_ROOT
        image = cv2.imread(str(PROJECT_ROOT / 'Image' / 'TestImage.png'))
        counts, unknown = read_cards_left_detailed(image)
        assert unknown == []
        assert counts['left'] == 14
