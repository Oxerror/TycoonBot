"""End to end: the status bar verifying tracked game state.

Uses the two real gameplay screenshots as two observations. Tracking
initialized from one screenshot must verify cleanly against the same
frame, and must flag mismatches against a frame from a different game.
"""

import cv2

from GameLogic.Card import Rank
from GameLogic.GameState import GameState
from ImageRecognition import PROJECT_ROOT
from StatusBarReader import read_status_bar

IMAGE_DIR = PROJECT_ROOT / 'Image'


def bar_from(image_name):
    image = cv2.imread(str(IMAGE_DIR / image_name))
    assert image is not None
    counts = read_status_bar(image)
    assert counts is not None
    return counts


def test_correct_tracking_verifies_clean():
    counts = bar_from('TestImage.png')
    state = GameState.from_status_bar(counts)
    assert state.verify_against(counts) == {}


def test_diverged_state_is_flagged():
    """State from one game checked against another: the bot 'messed up'."""
    state = GameState.from_status_bar(bar_from('TestImage.png'))
    mismatches = state.verify_against(bar_from('TestImage2.png'))

    assert mismatches
    # Spot-check one known difference: TestImage has one 3 unseen,
    # TestImage2 has none.
    assert mismatches[Rank.THREE] == (1, 0)


def test_bar_matches_opponents_card_counters():
    """The bar counts exactly the opponents' hidden cards.

    The opponents' "Cards Left" bubbles show 14+11+8 in TestImage and
    12+10+13 in TestImage2 — the bar totals must agree, confirming the
    unseen-cards interpretation GameState is built on.
    """
    assert GameState.from_status_bar(bar_from('TestImage.png')).total_unseen() == 33
    assert GameState.from_status_bar(bar_from('TestImage2.png')).total_unseen() == 35
