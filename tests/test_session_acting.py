"""Act mode presses each turn exactly once.

The live loop sees the same turn on several consecutive frames while
the game animates. Session must hand the plan to an act-mode executor
on the first of them only, and again only once the turn demonstrably
moved on (another player's arrow). Suggest mode is untouched by the
guard: it keeps recording a plan on every acting frame, which the
replay fixtures depend on.
"""

import numpy as np

from GameLogic.Card import Card, Rank
from InputExecutor import InputExecutor
from Session import TycoonSession

CONFIG = {
    'play_field': {'top': 0.4, 'bottom': 0.8, 'left': 0.3, 'right': 0.7},
    'hand_region': {'top': 0.75, 'bottom': 1.0, 'left': 0.0, 'right': 1.0},
}

FRAME = np.zeros((100, 100, 3), dtype=np.uint8)


class SceneReader:
    """A FrameReader stand-in describing one fixed table scene."""

    def __init__(self):
        self.active = 'player'
        self.hand_cards = [Card(Rank.FOUR), Card(Rank.FIVE)]

    def bar(self, frame):
        # One more unseen card than the opponents hold, so _suggest
        # sticks to the fast heuristic instead of the rollout search.
        counts = {rank: 0 for rank in Rank}
        counts.update({Rank.THREE: 1, Rank.FOUR: 1, Rank.FIVE: 2})
        return counts

    def counters(self, frame):
        return ({'player': len(self.hand_cards), 'left': 1, 'middle': 1,
                 'right': 1}, [], self.active, [])

    def field(self, play_field):
        return False, []

    def hand(self, hand_crop):
        return [], list(self.hand_cards)

    def revolution(self, frame):
        return False


class PressCounter:
    def __init__(self):
        self.sent = []

    def press_sequence(self, plan):
        self.sent.append(list(plan))


def acting_session():
    reader = SceneReader()
    pad = PressCounter()
    executor = InputExecutor(backend=pad, mode='act')
    session = TycoonSession(CONFIG, reader=reader, executor=executor)
    return session, reader, pad


def test_repeated_turn_frames_press_once():
    session, _, pad = acting_session()
    session.process_frame(FRAME)
    messages, _, _ = session.process_frame(FRAME)

    assert len(pad.sent) == 1
    assert any('already sent' in message for message in messages)


def test_presses_again_once_the_turn_moved_on():
    session, reader, pad = acting_session()
    session.process_frame(FRAME)

    reader.active = 'left'
    session.process_frame(FRAME)
    reader.active = 'player'
    session.process_frame(FRAME)

    assert len(pad.sent) == 2


def test_suggest_mode_still_plans_every_frame():
    reader = SceneReader()
    executor = InputExecutor()
    session = TycoonSession(CONFIG, reader=reader, executor=executor)
    session.process_frame(FRAME)
    session.process_frame(FRAME)

    assert len(executor.history) == 2
