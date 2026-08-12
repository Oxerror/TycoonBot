import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

PROJECT_ROOT = Path(__file__).parent


class CardCNN(nn.Module):
    """
    A simple CNN for recognizing card values and suits from images.
    """
    def __init__(self, num_classes):
        super(CardCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.25)

        # fc1 is created lazily on the first forward pass, once the flattened size is known
        self.fc1 = None
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 256).to(x.device)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class CardRecognizer:
    """
    Main class for card recognition using PyTorch.
    Supports both template matching (fast) and CNN-based recognition (more accurate).
    """
    def __init__(self, template_dir=None, model_path=None):
        if template_dir is None:
            template_dir = PROJECT_ROOT / 'Image' / 'templates'
        self.template_dir = Path(template_dir)
        self.templates = {}
        self.model = None
        self.class_names = []
        # (cache_key, template name) -> best matching scale, see template_match
        self._scale_cache = {}

        self._load_templates()

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def _load_templates(self):
        """Load template images for template matching."""
        if not self.template_dir.exists():
            print(f"Template directory not found: {self.template_dir}")
            return

        for template_file in self.template_dir.glob('*.png'):
            name = template_file.stem
            template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
            if template is None:
                continue
            self.templates[name] = template
            self.class_names.append(name)

        for template_file in self.template_dir.glob('*.PNG'):
            name = template_file.stem
            if name in self.templates:
                continue
            template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
            if template is None:
                continue
            self.templates[name] = template
            self.class_names.append(name)

        self.class_names = sorted(list(set(self.class_names)))
        print(f"Loaded {len(self.templates)} templates: {list(self.templates.keys())}")

    def apply_white_mask(self, image):
        """
        Apply white mask to isolate card symbols and numbers.
        Converts whiteish pixels to white, everything else to black.

        This improves recognition by:
        - Removing background variations (lighting, card texture)
        - Matching the black/white template format
        - Reducing noise from card artwork
        """
        if len(image.shape) == 3:
            lower_white = np.array([200, 200, 200], dtype=np.uint8)
            upper_white = np.array([255, 255, 255], dtype=np.uint8)
            white_mask = cv2.inRange(image, lower_white, upper_white)
        else:
            white_mask = cv2.inRange(image, 200, 255)

        result = np.zeros_like(image)
        if len(image.shape) == 3:
            result[white_mask == 255] = [255, 255, 255]
        else:
            result[white_mask == 255] = 255
        return result

    # On the player's turn the game dims every card that cannot join a
    # valid play to a uniform ~56%: white glyphs land at gray 142 while
    # staying neutral, and the card's red artwork stays saturated.
    # Measured on the recorded own-turn frames: greyed glyphs sit at
    # exactly 142 (thousands of pixels in [140, 149], the neighboring
    # bands nearly empty), bright glyphs at 255.
    DIM_GLYPH_BAND = (126, 170)
    DIM_NEUTRAL_SPREAD = 28

    def apply_dim_mask(self, image):
        """Isolate the glyphs of greyed-out (invalid-to-play) cards.

        The dim counterpart of apply_white_mask, keeping only the
        neutral-grey band the dimming maps white glyphs into. Hand
        reading runs both masks as separate passes: mixing the bands
        into one mask lets mid-grey card artwork distort the boxes of
        bright detections. The play field relies on the white mask to
        drop earlier (dimmed) tricks, so it must never use this one.
        """
        lo, hi = self.DIM_GLYPH_BAND
        result = np.zeros_like(image)
        if len(image.shape) == 3:
            channels = image.astype(np.int16)
            spread = channels.max(axis=2) - channels.min(axis=2)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dim = ((spread <= self.DIM_NEUTRAL_SPREAD)
                   & (gray >= lo) & (gray <= hi))
            result[dim] = (255, 255, 255)
        else:
            dim = (image >= lo) & (image <= hi)
            result[dim] = 255
        return result

    def _masked(self, image, apply_mask):
        """Resolve the apply_mask parameter: True for the white mask,
        'dim' for the greyed-out-glyph mask, False for the raw image."""
        if apply_mask == 'dim':
            return self.apply_dim_mask(image)
        if apply_mask:
            return self.apply_white_mask(image)
        return image

    def preprocess_image(self, image, target_size=(64, 64), apply_mask=True):
        """
        Preprocess image for CNN input.

        Args:
            image: BGR or grayscale image
            target_size: Target size for resizing
            apply_mask: Whether to apply white mask preprocessing

        Returns:
            Preprocessed tensor ready for model input
        """
        if apply_mask:
            image = self.apply_white_mask(image)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        resized = cv2.resize(gray, target_size)

        normalized = resized.astype(np.float32) / 255.0

        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(device)

    @staticmethod
    def _glyph_class(name):
        """Group templates whose on-screen glyphs share a size."""
        if name in ('Heart', 'Diamond', 'Spade', 'Cross'):
            return 'suit'
        if name in ('Joker', 'Wonder'):
            return 'special'
        return 'rank'

    @staticmethod
    def _rotate_template(template, angle):
        """Rotate a template around its center, expanding the canvas so nothing is cut off."""
        h, w = template.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        matrix[0, 2] += new_w / 2 - w / 2
        matrix[1, 2] += new_h / 2 - h / 2
        return cv2.warpAffine(template, matrix, (new_w, new_h), borderValue=0)

    def template_match(self, image, threshold=0.7, apply_mask=True,
                       scales=None, angles=(0,), downscale=1, cache_key=None):
        """
        Perform template matching to find cards in the image.

        Args:
            image: Input image (BGR or grayscale)
            threshold: Matching threshold (0-1)
            apply_mask: True for the white mask, 'dim' for the
                greyed-out-glyph mask, False for none
            scales: Iterable of template scales to try. The templates were
                cropped at inconsistent sizes, so the step must be fine
                enough that every template hits its true on-screen scale.
            angles: Rotation angles (degrees) to try per template. The hand
                is displayed as a fan, so cards near the edges are rotated
                up to ~20 degrees.
            downscale: Integer factor to shrink the search image by before
                matching. 2 is ~16x faster and still reliable for the large
                card glyphs; reported locations are in original coordinates.
            cache_key: When set (e.g. 'hand'), the best matching scale per
                template is remembered under this key and later calls only
                sweep a small neighborhood around it. On-screen glyph sizes
                are constant for a given resolution, so this is safe and
                collapses the scale sweep after each template has been seen
                confidently once.

        Returns:
            List of (name, confidence, location) tuples
        """
        if scales is None:
            scales = np.arange(0.4, 1.61, 0.05)

        image = self._masked(image, apply_mask)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if downscale > 1:
            gray = cv2.resize(gray, (gray.shape[1] // downscale, gray.shape[0] // downscale))

        results = []

        # On-screen glyph heights are consistent within a class (all rank
        # glyphs are ~the same size), so templates already pinned by the
        # cache give unseen templates of the same class a strong prior.
        prior_heights = {}
        if cache_key:
            for (key, name), scale in self._scale_cache.items():
                if key == cache_key and name in self.templates:
                    glyph_class = self._glyph_class(name)
                    prior_heights.setdefault(glyph_class, []).append(
                        self.templates[name].shape[0] * scale)

        for name, template in self.templates.items():
            cached_scale = self._scale_cache.get((cache_key, name)) if cache_key else None
            if cached_scale is not None:
                trial_scales = (cached_scale,)
            else:
                heights = prior_heights.get(self._glyph_class(name))
                if heights:
                    center = float(np.median(heights)) / template.shape[0]
                    trial_scales = tuple(center + step for step in
                                         (-0.10, -0.05, 0.0, 0.05, 0.10))
                else:
                    trial_scales = scales

            best_confidence, best_scale = 0.0, None

            for angle in angles:
                rotated = self._rotate_template(template, angle) if angle else template

                for scale in trial_scales:
                    h, w = rotated.shape
                    new_h, new_w = int(h * scale / downscale), int(w * scale / downscale)

                    if new_h > gray.shape[0] or new_w > gray.shape[1]:
                        continue

                    if new_h < 10 or new_w < 10:
                        continue

                    scaled_template = cv2.resize(rotated, (new_w, new_h))

                    result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)

                    locations = np.where(result >= threshold)

                    for pt in zip(*locations[::-1]):
                        confidence = result[pt[1], pt[0]]
                        if confidence > best_confidence:
                            best_confidence, best_scale = float(confidence), scale
                        results.append({
                            'name': name,
                            'confidence': float(confidence),
                            'location': (pt[0] * downscale, pt[1] * downscale,
                                         new_w * downscale, new_h * downscale),
                            'angle': angle,
                        })

            # Only very confident matches may pin the scale, so a noisy
            # false positive cannot lock future sweeps to a wrong size.
            if cache_key and cached_scale is None and best_confidence >= 0.85:
                self._scale_cache[(cache_key, name)] = best_scale

        results = self._non_max_suppression(results)

        return results

    SUIT_NAMES = ('Heart', 'Diamond', 'Spade', 'Cross')

    def refine_suit_detections(self, image, detections, apply_mask=True):
        """
        Re-classify suit detections at full resolution.

        The scale-swept search runs downscaled, where suit glyphs are
        ~13px and Spade/Cross (and their mirrored artwork) blur into
        each other. Re-matching all four suit templates inside each
        detection's window at full resolution picks the right one.
        Mutates and returns the detections. Pass the same apply_mask
        the detections were matched with.
        """
        image = self._masked(image, apply_mask)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        for detection in detections:
            if detection['name'] not in self.SUIT_NAMES:
                continue

            x, y, w, h = detection['location']
            angle = detection.get('angle', 0)
            pad = max(4, w // 4)
            window = gray[max(0, y - pad):y + h + pad,
                          max(0, x - pad):x + w + pad]

            best_name, best_score = detection['name'], -1.0
            for suit in self.SUIT_NAMES:
                rotated = self._rotate_template(self.templates[suit], angle) \
                    if angle else self.templates[suit]

                for factor in (0.9, 1.0, 1.1):
                    new_w, new_h = int(w * factor), int(h * factor)
                    if new_w < 5 or new_h < 5:
                        continue
                    if new_h > window.shape[0] or new_w > window.shape[1]:
                        continue

                    scaled = cv2.resize(rotated, (new_w, new_h))
                    score = float(cv2.matchTemplate(
                        window, scaled, cv2.TM_CCOEFF_NORMED).max())
                    if score > best_score:
                        best_name, best_score = suit, score

            detection['name'] = best_name
            detection['confidence'] = best_score

        return detections

    @staticmethod
    def _boxes_conflict(location_a, location_b, iou_threshold=0.5):
        """Whether two detection boxes describe the same glyph."""
        x1, y1, w1, h1 = location_a
        x2, y2, w2, h2 = location_b

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return False
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        iou = intersection / union
        # Same glyph matched at two scales gives nested boxes whose
        # IoU stays low; suppress by smaller-box coverage as well.
        smaller_area = min(w1 * h1, w2 * h2)
        coverage = intersection / smaller_area

        return iou > iou_threshold or coverage > 0.7

    def _non_max_suppression(self, detections, iou_threshold=0.5):
        """Remove overlapping detections."""
        if not detections:
            return []

        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        kept = []
        for detection in detections:
            if not any(self._boxes_conflict(detection['location'],
                                            kept_det['location'], iou_threshold)
                       for kept_det in kept):
                kept.append(detection)

        return kept

    def create_model(self, num_classes=None):
        """Create a new CNN model."""
        if num_classes is None:
            num_classes = len(self.class_names)

        self.model = CardCNN(num_classes).to(device)
        return self.model

    def save_model(self, path):
        """Save the trained model."""
        if self.model is None:
            print("No model to save!")
            return

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load a trained model."""
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            return False

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.class_names = checkpoint['class_names']

        self.model = CardCNN(len(self.class_names)).to(device)

        # Run a dummy forward pass so the lazily-created fc1 layer exists before loading weights
        dummy_input = torch.zeros(1, 1, 64, 64).to(device)
        with torch.no_grad():
            self.model(dummy_input)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded from {path}")
        return True

    def predict_cnn(self, image):
        """
        Predict using the CNN model.

        Args:
            image: Input image

        Returns:
            Predicted class name and confidence
        """
        if self.model is None:
            print("No CNN model loaded. Using template matching instead.")
            return None

        self.model.eval()

        with torch.no_grad():
            tensor = self.preprocess_image(image)
            output = self.model(tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            class_name = self.class_names[predicted.item()]

            return {
                'name': class_name,
                'confidence': confidence.item(),
            }

    def recognize(self, image, method='template', threshold=0.7):
        """
        Main recognition method.

        Args:
            image: Input image
            method: 'template' for template matching, 'cnn' for CNN
            threshold: Confidence threshold

        Returns:
            List of detected cards
        """
        if method == 'cnn' and self.model is not None:
            return self.predict_cnn(image)

        return self.template_match(image, threshold)


_recognizer = None


def get_recognizer():
    """Get or create the global recognizer instance."""
    global _recognizer
    if _recognizer is None:
        _recognizer = CardRecognizer()
    return _recognizer


def recognizeWithTemplates(image, threshold=0.7, apply_mask=True):
    """
    Recognize cards in an image using template matching.
    This is the main function called from VideoStream.py

    Args:
        image: Input image (BGR format from OpenCV)
        threshold: Matching threshold
        apply_mask: Whether to apply white mask preprocessing (default True)

    Returns:
        List of detected card names
    """
    recognizer = get_recognizer()
    results = recognizer.template_match(image, threshold=threshold, apply_mask=apply_mask)

    if isinstance(results, list):
        return [r['name'] for r in results]
    elif isinstance(results, dict):
        return [results['name']]
    return []


# Parameters tuned on the committed gameplay screenshots: the hand fan
# rotates cards up to ~20 degrees and the templates were cropped at
# inconsistent sizes, so matching needs an angle sweep and a fine scale
# sweep; downscale keeps that affordable.
HAND_MATCH_PARAMS = {
    'threshold': 0.75,
    'angles': range(-20, 21, 5),
    'downscale': 2,
    'cache_key': 'hand',
}


# Cards are tossed onto the table with up to ~20 degrees of rotation.
# The game dims previous tricks, so the white mask naturally isolates
# the current (bright) trick — older plays fall below the mask threshold.
# No cache_key: unlike the hand, glyph sizes on the table vary from card
# to card, so pinning one scale per template makes results depend on
# what happened to be seen first.
FIELD_MATCH_PARAMS = {
    'threshold': 0.75,
    'angles': range(-20, 21, 5),
    'downscale': 2,
}


def read_hand_detections(image):
    """
    Raw hand detections: a bright pass plus a greyed-out-cards pass.

    On the player's turn the game dims every card that cannot join a
    valid play; the white mask erases those glyphs wholesale, so a
    second pass matches inside the dim band they land in instead. The
    bright pass runs exactly as always and its detections win every
    overlap, so hands without greyed-out cards read as before; the dim
    pass only fills the gaps. Card artwork also has mid-grey pixels,
    so the dim pass runs on most non-empty crops; its stray artwork
    matches die in the rank-suit pairing, which keeps the merge safe
    on bright hands.

    Args:
        image: BGR image of the player's hand area

    Returns:
        List of detection dicts, as CardRecognizer.template_match.
    """
    recognizer = get_recognizer()
    detections = recognizer.template_match(image, **HAND_MATCH_PARAMS)
    recognizer.refine_suit_detections(image, detections)

    dim_mask = recognizer.apply_dim_mask(image)
    dim_plane = dim_mask[:, :, 0] if dim_mask.ndim == 3 else dim_mask
    # Skipping truly empty crops (hand played out) is the only safe
    # shortcut: a single greyed-out card's glyph load is not separable
    # from ordinary artwork grey by pixel count alone.
    if cv2.countNonZero(dim_plane) > 500:
        dim_detections = recognizer.template_match(image, apply_mask='dim',
                                                   **HAND_MATCH_PARAMS)
        recognizer.refine_suit_detections(image, dim_detections,
                                          apply_mask='dim')
        detections.extend(
            dim for dim in dim_detections
            if not any(CardRecognizer._boxes_conflict(dim['location'],
                                                      bright['location'])
                       for bright in detections))
    return detections


def read_hand(image):
    """
    Recognize the cards in a hand-region screenshot.

    Args:
        image: BGR image of the player's hand area

    Returns:
        List of GameLogic Card objects, sorted left to right
    """
    from GameLogic.HandReader import detections_to_cards

    return detections_to_cards(read_hand_detections(image))


# The persistent "Revolution - Flip Strength" badge shown above the
# player box while a revolution is active. Detected via its red arrow,
# matched as a binary red-mask shape (red also appears in character
# panels, so shape matters, not just color).
REVOLUTION_TEMPLATE_PATH = PROJECT_ROOT / 'Image' / 'templates' / 'indicators' / 'revolution.png'
REVOLUTION_REGION = {'top': 0.44, 'bottom': 0.64, 'left': 0.03, 'right': 0.26}
REVOLUTION_THRESHOLD = 0.6

_revolution_template = None


def read_revolution_indicator(frame):
    """True while the Revolution badge is visible on a full game frame."""
    global _revolution_template
    if _revolution_template is None:
        _revolution_template = cv2.imread(str(REVOLUTION_TEMPLATE_PATH),
                                          cv2.IMREAD_GRAYSCALE)
        if _revolution_template is None:
            raise FileNotFoundError(f"Missing {REVOLUTION_TEMPLATE_PATH}")

    height, width = frame.shape[:2]
    region = frame[int(height * REVOLUTION_REGION['top']):int(height * REVOLUTION_REGION['bottom']),
                   int(width * REVOLUTION_REGION['left']):int(width * REVOLUTION_REGION['right'])]

    channels = region.astype(np.int16)
    red_mask = ((channels[:, :, 2] - np.maximum(channels[:, :, 0], channels[:, :, 1]) > 60)
                & (channels[:, :, 2] > 120)).astype(np.uint8) * 255

    scale = height / 1080
    template = _revolution_template
    if scale != 1.0:
        template = cv2.resize(template, (max(1, int(template.shape[1] * scale)),
                                         max(1, int(template.shape[0] * scale))))
    if (template.shape[0] > red_mask.shape[0]
            or template.shape[1] > red_mask.shape[1]):
        return False

    score = cv2.matchTemplate(red_mask, template, cv2.TM_CCOEFF_NORMED).max()
    return float(score) >= REVOLUTION_THRESHOLD


# Event banners (All Pass, 8 Stop, Done, Game Set) cover the play field
# with large white shapes that fool the card templates — the Done star
# reads as a Wonder, the 8 Stop banner contains a giant 8. Measured
# white coverage: normal fields stay below 0.03, banners exceed 0.12.
BANNER_WHITE_FRACTION = 0.07


def banner_visible(field_image):
    """True when an event banner covers the play-field crop."""
    channels = field_image.astype(np.int16)
    white = ((channels[:, :, 0] > 230) & (channels[:, :, 1] > 230)
             & (channels[:, :, 2] > 230))
    fraction = float(white.sum()) / (field_image.shape[0] * field_image.shape[1])
    return fraction > BANNER_WHITE_FRACTION


def read_play_field(image):
    """
    Recognize the cards of the current trick on the table.

    Only the bright (most recent) trick survives the white mask; the
    game dims earlier plays, so those drop out on their own.

    Args:
        image: BGR image of the play-field area

    Returns:
        List of GameLogic Card objects, sorted left to right. A card
        whose suit symbol is covered by a neighbor still counts (with
        suit None): dropping it would understate the trick size, and
        the required play size matters more than the suit.
    """
    from GameLogic.HandReader import detections_to_cards

    recognizer = get_recognizer()
    detections = recognizer.template_match(image, **FIELD_MATCH_PARAMS)
    recognizer.refine_suit_detections(image, detections)
    return detections_to_cards(detections, keep_unpaired_ranks=True)


def recognizeWithCNN(image):
    """
    Recognize a single card using CNN.

    Args:
        image: Input image containing a single card

    Returns:
        Predicted card name
    """
    recognizer = get_recognizer()
    result = recognizer.recognize(image, method='cnn')

    if result:
        return result['name']
    return None


def test_with_visualization(image_path=None, threshold=0.7, method='template'):
    """
    Test recognition on images and display results with bounding boxes.

    Args:
        image_path: Path to a specific image, or None to test all images in Image/
        threshold: Detection threshold
        method: 'template' or 'cnn'
    """
    recognizer = CardRecognizer(model_path='card_model.pth')

    if image_path:
        test_images = [Path(image_path)]
    else:
        image_dir = Path('Image')
        test_images = list(image_dir.glob('TestImage*.png')) + list(image_dir.glob('TestImage*.PNG'))
        # Windows globbing is case-insensitive, so both patterns can match the same file
        test_images = sorted(set(test_images))

    if not test_images:
        print("No test images found!")
        return

    print(f"Found {len(test_images)} test image(s)")
    print(f"Using method: {method}")
    print('-' * 40)

    colors = (
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    )

    for img_path in test_images:
        print(f"\nProcessing: {img_path}")

        image = cv2.imread(str(img_path))
        if image is None:
            print("  Failed to load image!")
            continue

        output = image.copy()

        if method == 'template':
            results = recognizer.template_match(image, threshold)

            print(f"  Found {len(results)} detection(s):")

            for i, detection in enumerate(results):
                name = detection['name']
                confidence = detection['confidence']
                x, y, w, h = detection['location']

                color = colors[i % len(colors)]

                cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

                label = f"{name} ({confidence:.0%})"
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                cv2.rectangle(output, (x, y - label_h - 10),
                              (x + label_w + 4, y), color, -1)

                cv2.putText(output, label, (x + 2, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                print(f"    [{i + 1}] {name}: {confidence:.1%} at ({x}, {y})")
        else:
            result = recognizer.predict_cnn(image)
            if result:
                name = result['name']
                confidence = result['confidence']
                print(f"  CNN Prediction: {name} ({confidence:.1%})")

                label = f"CNN: {name} ({confidence:.0%})"
                cv2.putText(output, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        window_name = f"Detection: {img_path.name}"
        cv2.imshow(window_name, output)

        output_path = img_path.parent / f"{img_path.stem}_detected{img_path.suffix}"
        cv2.imwrite(str(output_path), output)
        print(f"  Saved result to: {output_path}")

    print('\n' + '-' * 40)
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        threshold = 0.8
        method = 'template'
        image_path = None

        for arg in sys.argv[2:]:
            if arg.startswith('--threshold='):
                threshold = float(arg.split('=')[1])
            elif arg.startswith('--method='):
                method = arg.split('=')[1]
            elif arg.startswith('--image='):
                image_path = arg.split('=')[1]

        test_with_visualization(image_path, threshold, method)
    else:
        print("PyTorch Image Recognition Module")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        recognizer = CardRecognizer()
        print(f"\nClass names: {recognizer.class_names}")

        if recognizer.class_names:
            model = recognizer.create_model()
            print(f"\nModel created with {len(recognizer.class_names)} classes")
            print(model)

        print('\n' + '=' * 50)
        print("To run visual test on TestImages:")
        print("  python ImageRecognition.py test")
        print("\nOptions:")
        print("  --threshold=0.8    Set detection threshold")
        print("  --method=cnn       Use CNN instead of template matching")
        print("  --image=path.png   Test a specific image")
