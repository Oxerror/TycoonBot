"""The full dry-run loop over the recorded session.

Replays every captured frame through TycoonSession — read -> track ->
search -> plan inputs — and pins the button sequence planned on each
acting frame. The sequences are fixtured like the recognition
readings: recorded beside the (gitignored) captures on the first run,
compared exactly on every later one, so any change to recognition,
tracking, search, or the planner that alters what the bot would press
shows up as a diff. Delete Image/captures/input_plans.json to accept a
new baseline.

Skips when no captures exist (fresh clone); the recognition sidecars
make the replay a matter of seconds, but the search still runs for
real on every acting frame, hence the slow mark.
"""

import json

import cv2
import pytest

from FrameReader import CachedFrameReader
from ImageRecognition import PROJECT_ROOT
from ScreenCapture import loadConfig
from Session import TycoonSession

CAPTURE_DIR = PROJECT_ROOT / 'Image' / 'captures'
FIXTURE = CAPTURE_DIR / 'input_plans.json'


@pytest.mark.slow
def test_planned_inputs_match_the_fixture():
    frames = sorted(CAPTURE_DIR.glob('*.png'))
    if not frames:
        pytest.skip("no recorded captures on this machine")

    reader = CachedFrameReader()
    session = TycoonSession(loadConfig(), reader=reader)

    plans = {}
    for path in frames:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        reader.begin_frame(path)
        before = len(session.executor.history)
        session.process_frame(frame)
        for plan in session.executor.history[before:]:
            plans[path.name] = plan

    assert plans, "the recorded session never reached an own turn"

    if not FIXTURE.exists():
        FIXTURE.write_text(json.dumps(plans, indent=1))
        pytest.skip(f"fixture recorded ({len(plans)} acting frame(s)); "
                    "rerun to compare")

    assert plans == json.loads(FIXTURE.read_text())
