"""Collect gameplay screenshots while you play.

Run this alongside the game to gather real frames for tests and
template extraction:

    python CaptureData.py

Frames are saved to Image/captures/ (gitignored — promote good ones to
Image/ by hand) together with a .json sidecar holding what the cheap
readers saw at capture time, so promoted frames arrive pre-labeled.

A frame is saved when:
- the status bar reading changed since the last save (one frame per
  play, the moments with fresh tricks and changed counters),
- a "Cards Left" counter shows a digit the template set does not know
  yet (5/6/7/9 are still missing) — these frames are exactly what is
  needed to complete the digit templates,
- or nothing was saved for `interval_seconds` (round transitions and
  other screens where the bar is not visible).

Every frame has the configured redact_regions blacked out BEFORE it is
written, so the platform user id overlay never reaches the disk. The
loop only runs the cheap readers (no torch, no card recognition), so it
will not slow the game down.
"""

import json
import time
from datetime import datetime

import cv2
import mss

from CardsLeftReader import read_cards_left_detailed
from ScreenCapture import PROJECT_ROOT, applyRedactions, getScreen, loadConfig
from StatusBarReader import read_status_bar

CAPTURE_DIR = PROJECT_ROOT / 'Image' / 'captures'

DEFAULT_CAPTURE_SETTINGS = {
    # Save at least one frame this often even without a trigger.
    'interval_seconds': 45,
    # Unknown-digit frames fire at most this often.
    'unknown_digit_cooldown': 20,
    # Stop after this many frames so a forgotten session cannot fill the disk.
    'max_frames': 200,
    # Seconds between screen grabs.
    'poll_seconds': 0.5,
}


def frames_differ(frame_a, frame_b, threshold=2.0):
    """True when two frames differ noticeably (mean abs diff on small grayscale)."""
    if frame_a is None or frame_b is None:
        return True
    if frame_a.shape != frame_b.shape:
        return True

    def small_gray(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (160, 90)).astype('int16')

    diff = abs(small_gray(frame_a) - small_gray(frame_b)).mean()
    return diff > threshold


def save_frame(frame, reason, bar_counts, cards_left, unknown_digits,
               active_player=None):
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = f"{stamp}_{reason}"
    path = CAPTURE_DIR / f"{base}.png"
    suffix = 1
    while path.exists():
        path = CAPTURE_DIR / f"{base}_{suffix}.png"
        suffix += 1

    cv2.imwrite(str(path), frame)

    sidecar = {
        'captured_at': datetime.now().isoformat(timespec='seconds'),
        'reason': reason,
        'status_bar': {rank.name: count for rank, count in bar_counts.items()}
                      if bar_counts is not None else None,
        'cards_left': cards_left,
        'unknown_digits': unknown_digits,
        'active_player': active_player,
    }
    path.with_suffix('.json').write_text(json.dumps(sidecar, indent=2))

    return path


def captureLoop():
    config = loadConfig()
    settings = dict(DEFAULT_CAPTURE_SETTINGS)
    settings.update(config.get('capture', {}))

    sct = mss.mss()

    last_saved_bar = None
    last_saved_frame = None
    last_save_time = 0.0
    last_unknown_save_time = 0.0
    saved = 0

    print(f"Capturing to {CAPTURE_DIR}")
    print(f"Stops after {settings['max_frames']} frames; Ctrl+C to stop earlier.")

    try:
        while saved < settings['max_frames']:
            frame = getScreen(sct, config['monitor'])
            applyRedactions(frame, config['redact_regions'])

            now = time.monotonic()
            bar_counts = read_status_bar(frame)
            cards_left, unknown_digits, active_player = read_cards_left_detailed(frame)

            reason = None
            if bar_counts is not None and bar_counts != last_saved_bar:
                reason = 'play'
            elif unknown_digits and now - last_unknown_save_time >= settings['unknown_digit_cooldown']:
                reason = 'unknown-digit'
            elif now - last_save_time >= settings['interval_seconds']:
                reason = 'interval'

            if reason and frames_differ(frame, last_saved_frame):
                path = save_frame(frame, reason, bar_counts, cards_left,
                                  unknown_digits, active_player)
                saved += 1
                last_saved_bar = bar_counts
                last_saved_frame = frame
                last_save_time = now
                if reason == 'unknown-digit':
                    last_unknown_save_time = now
                print(f"[{saved}/{settings['max_frames']}] {reason}: {path.name}"
                      + (f" (unknown digit at {', '.join(unknown_digits)})"
                         if unknown_digits else ""))

            time.sleep(settings['poll_seconds'])
    except KeyboardInterrupt:
        pass

    print(f"Done: {saved} frame(s) in {CAPTURE_DIR}")


if __name__ == "__main__":
    captureLoop()
