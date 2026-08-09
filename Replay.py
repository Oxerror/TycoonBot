"""Replay captured frames through the game-following logic.

The frames in Image/captures/ are a chronologically ordered recording
of a real session (one frame per play), so feeding them through the
same TycoonSession the live loop uses validates the whole pipeline —
play tracking, alarms, round transitions, turn detection — without
having to play:

    python Replay.py [glob]

Every ALARM/WARNING line is a finding worth investigating.

Recognition results are cached in a *.readings.json sidecar next to
each frame (frames never change once captured), so only the first
replay pays for template matching — later ones take seconds. Delete
the sidecars to force fresh recognition.
"""

import glob
import sys
from pathlib import Path

import cv2

from FrameReader import CachedFrameReader
from ScreenCapture import loadConfig
from Session import TycoonSession


def replay(pattern):
    reader = CachedFrameReader()
    session = TycoonSession(loadConfig(), reader=reader)

    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No frames match {pattern}")
        return

    issues = 0
    for path in files:
        frame = cv2.imread(path)
        if frame is None:
            continue
        reader.begin_frame(path)
        messages, _, _ = session.process_frame(frame)
        print(f"=== {Path(path).name}")
        for message in messages:
            print(f"    {message}")
            if message.startswith(('ALARM', 'WARNING')):
                issues += 1

    print(f"\n{len(files)} frame(s) replayed, {issues} ALARM/WARNING line(s)")


if __name__ == "__main__":
    replay(sys.argv[1] if len(sys.argv) > 1 else 'Image/captures/*.png')
