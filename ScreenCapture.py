"""Shared screen-grabbing helpers.

Used by both VideoStream (the live recognition loop) and CaptureData
(the training-data collector). Kept free of heavy imports so the
capture tool starts fast and stays cheap.
"""

import json
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / 'config.json'

DEFAULT_CONFIG = {
    # mss monitor index: 0 is all monitors combined, 1 the primary,
    # 2+ the additional monitors.
    'monitor': 2,
    # Regions as fractions of the captured frame (top/bottom/left/right).
    'hand_region': {'top': 0.75, 'bottom': 1.0, 'left': 0.0, 'right': 1.0},
    'play_field': {'top': 0.4, 'bottom': 0.8, 'left': 0.333, 'right': 0.667},
    # Screen areas blacked out on every captured frame BEFORE anything
    # is stored or displayed — e.g. the platform user id overlay in the
    # bottom-right corner. Solid black is irreversible (unlike blur)
    # and reads as background for the recognition mask.
    'redact_regions': [
        {'top': 0.96, 'bottom': 1.0, 'left': 0.853, 'right': 1.0},
    ],
}


def loadConfig():
    config = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            config.update(json.loads(CONFIG_PATH.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Could not read {CONFIG_PATH.name} ({e}), using defaults")
    return config


def cropRegion(frame, region):
    height, width = frame.shape[:2]
    top = int(height * region['top'])
    bottom = int(height * region['bottom'])
    left = int(width * region['left'])
    right = int(width * region['right'])
    return frame[top:bottom, left:right]


def applyRedactions(frame, regions):
    """Black out the configured regions in place and return the frame."""
    height, width = frame.shape[:2]
    for region in regions:
        top = int(height * region['top'])
        bottom = int(height * region['bottom'])
        left = int(width * region['left'])
        right = int(width * region['right'])
        frame[top:bottom, left:right] = 0
    return frame


def getScreen(sct, monitor_index):
    """Grab one frame from the configured monitor as a BGR image."""
    if monitor_index >= len(sct.monitors):
        fallback = min(1, len(sct.monitors) - 1)
        print(f"Monitor {monitor_index} not found "
              f"({len(sct.monitors) - 1} monitor(s) available), using monitor {fallback}")
        monitor_index = fallback

    monitor = sct.monitors[monitor_index]
    img = np.array(sct.grab(monitor))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
