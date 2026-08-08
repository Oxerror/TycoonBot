import json
import time
from pathlib import Path

import cv2
import mss
import numpy as np

from ImageRecognition import HAND_MATCH_PARAMS, get_recognizer, read_play_field
from GameLogic.GameState import GameState
from GameLogic.HandReader import detections_to_cards
from GameLogic.PlayTracker import PlayTracker
from CardsLeftReader import read_cards_left
from StatusBarReader import read_status_bar

PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / 'config.json'

DEFAULT_CONFIG = {
    # mss monitor index: 0 is all monitors combined, 1 the primary,
    # 2+ the additional monitors.
    'monitor': 2,
    # Regions as fractions of the captured frame (top/bottom/left/right).
    'hand_region': {'top': 0.75, 'bottom': 1.0, 'left': 0.0, 'right': 1.0},
    'play_field': {'top': 0.4, 'bottom': 0.8, 'left': 0.333, 'right': 0.667},
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


def drawDetections(image, detections):
    """Draw bounding boxes and labels for detected cards."""
    output = image.copy()

    # Colors for different card types (BGR format)
    colors = {
        # Numbers
        '2': (0, 255, 0), '3': (0, 255, 0), '4': (0, 255, 0), '5': (0, 255, 0),
        '6': (0, 255, 0), '7': (0, 255, 0), '8': (0, 255, 0), '9': (0, 255, 0), '10': (0, 255, 0),
        # Face cards
        'Jack': (255, 0, 255), 'Queen': (255, 0, 255), 'King': (255, 0, 255), 'Ace': (255, 0, 255),
        # Suits
        'Heart': (0, 0, 255), 'Diamond': (0, 0, 255),
        'Spade': (255, 255, 0), 'Cross': (255, 255, 0),
        # Special
        'Joker': (0, 255, 255), 'Wonder': (0, 255, 255),
    }
    default_color = (128, 128, 128)

    for detection in detections:
        name = detection['name']
        confidence = detection['confidence']
        x, y, w, h = detection['location']

        color = colors.get(name, default_color)

        # Draw bounding box
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        # Draw label background
        label = f"{name} ({confidence:.0%})"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(output, (x, y - label_h - 6), (x + label_w + 4, y), color, -1)

        # Draw label text
        cv2.putText(output, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return output


def videoCapturing():
    config = loadConfig()
    recognizer = get_recognizer()
    sct = mss.mss()

    # Tracked state, initialized from the first readable status bar and
    # kept current by observed plays. The bar is only read as ground
    # truth: a persistent divergence means the bot messed up.
    game_state = None
    tracker = None
    diverged_frames = 0

    print("Capturing... press 'q' in a window to quit.")

    while True:
        frame = getScreen(sct, config['monitor'])

        playField = cropRegion(frame, config['play_field'])
        currentHand = cropRegion(frame, config['hand_region'])

        start = time.time()
        detections = recognizer.template_match(currentHand, **HAND_MATCH_PARAMS)
        cards = detections_to_cards(detections)
        trick = read_play_field(playField)
        elapsed = time.time() - start

        handWithDetections = drawDetections(currentHand, detections)

        print(f"Hand ({elapsed:.1f}s): {cards}")
        if trick:
            print(f"Current trick: {trick}")

        bar_counts = read_status_bar(frame)
        if bar_counts is None:
            print("Status bar: not visible")
        elif game_state is None:
            game_state = GameState.from_status_bar(bar_counts)
            tracker = PlayTracker(game_state)
            tracker.update(trick, cards)
            print(f"Tracking started: {game_state.total_unseen()} unseen cards")
        else:
            try:
                for event in tracker.update(trick, cards):
                    who = 'we' if event['by_player'] else 'opponent'
                    print(f"Play observed ({who}): {event['cards']}")
            except ValueError as error:
                print(f"ALARM: impossible play observed - {error}")

            mismatches = game_state.verify_against(bar_counts)
            if not mismatches:
                diverged_frames = 0
                print("State verified: tracking matches the game")
            else:
                # One divergent frame can be a play caught mid-animation;
                # a persistent one means the bot lost track of the game.
                diverged_frames += 1
                diff = ', '.join(f"{r.name} {t}->{a}"
                                 for r, (t, a) in mismatches.items())
                if diverged_frames >= 2:
                    print(f"ALARM: bot state diverged from the game ({diff})")
                    print("Re-syncing from the status bar.")
                    game_state = GameState.from_status_bar(bar_counts)
                    tracker = PlayTracker(game_state)
                    tracker.update(trick, cards)
                    diverged_frames = 0
                else:
                    print(f"State mismatch this frame ({diff}), waiting one frame")

            counters = read_cards_left(frame)
            opponent_counts = [counters[k] for k in ('left', 'middle', 'right')]
            if None not in opponent_counts:
                opponents_total = sum(opponent_counts)
                if opponents_total != game_state.total_unseen():
                    print(f"Cards-left cross-check: bubbles say {opponents_total}, "
                          f"tracking says {game_state.total_unseen()}")

        cv2.imshow('Field', playField)
        cv2.imshow('Hand', handWithDetections)

        # imshow windows stay responsive only while waitKey pumps events
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    videoCapturing()
