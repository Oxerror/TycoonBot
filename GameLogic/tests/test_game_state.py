import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from GameLogic.Card import Card, Rank, Suit
from GameLogic.GameState import GameState


def full_counts(**overrides):
    counts = {rank: 4 for rank in Rank}
    counts[Rank.JOKER] = 2
    counts[Rank.WONDER] = 1
    for name, value in overrides.items():
        counts[Rank[name]] = value
    return counts


class TestConstruction:
    def test_from_status_bar(self):
        state = GameState.from_status_bar(full_counts())
        assert state.unseen[Rank.KING] == 4
        assert state.unseen[Rank.WONDER] == 1

    def test_missing_rank_rejected(self):
        counts = full_counts()
        del counts[Rank.KING]
        with pytest.raises(ValueError, match='KING'):
            GameState(counts)

    def test_total_unseen(self):
        assert GameState(full_counts()).total_unseen() == 13 * 4 + 2 + 1


class TestObservingPlays:
    def test_play_decrements_rank(self):
        state = GameState(full_counts())
        state.observe_opponent_play([Card(Rank.KING, Suit.HEARTS)])
        assert state.unseen[Rank.KING] == 3

    def test_pair_decrements_twice(self):
        state = GameState(full_counts())
        state.observe_opponent_play([
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.KING, Suit.SPADES),
        ])
        assert state.unseen[Rank.KING] == 2

    def test_impossible_play_raises(self):
        state = GameState(full_counts(KING=0))
        with pytest.raises(ValueError, match='KING'):
            state.observe_opponent_play([Card(Rank.KING, Suit.HEARTS)])


class TestVerification:
    def test_matching_state_verifies_clean(self):
        state = GameState(full_counts())
        assert state.verify_against(full_counts()) == {}

    def test_tracked_play_still_verifies(self):
        """After the bot tracks a play, it must match the updated bar."""
        state = GameState(full_counts())
        state.observe_opponent_play([Card(Rank.ACE, Suit.SPADES)])
        assert state.verify_against(full_counts(ACE=3)) == {}

    def test_missed_play_is_detected(self):
        """The bot missed a play: bar moved on, tracking did not."""
        state = GameState(full_counts())
        mismatches = state.verify_against(full_counts(ACE=3, KING=2))
        assert mismatches == {
            Rank.ACE: (4, 3),
            Rank.KING: (4, 2),
        }

    def test_unreadable_bar_returns_none(self):
        state = GameState(full_counts())
        assert state.verify_against(None) is None
