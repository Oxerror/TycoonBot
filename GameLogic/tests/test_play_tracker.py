import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from GameLogic.Card import Card, Rank, Suit
from GameLogic.GameState import GameState
from GameLogic.PlayTracker import PlayTracker


def full_counts(**overrides):
    counts = {rank: 4 for rank in Rank}
    counts[Rank.JOKER] = 2
    counts[Rank.WONDER] = 1
    for name, value in overrides.items():
        counts[Rank[name]] = value
    return counts


HAND = [Card(Rank.FIVE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]


def make_tracker(**overrides):
    state = GameState(full_counts(**overrides))
    return PlayTracker(state), state


class TestBaseline:
    def test_first_frame_reports_nothing(self):
        tracker, state = make_tracker()
        events = tracker.update([Card(Rank.NINE, Suit.CLUBS)], HAND)
        assert events == []
        assert state.unseen[Rank.NINE] == 4

    def test_pre_existing_trick_never_becomes_a_play(self):
        tracker, state = make_tracker()
        trick = [Card(Rank.NINE, Suit.CLUBS)]
        tracker.update(trick, HAND)
        events = tracker.update(trick, HAND)
        assert events == []
        assert state.unseen[Rank.NINE] == 4


class TestOpponentPlays:
    def test_new_card_is_an_opponent_play(self):
        tracker, state = make_tracker()
        tracker.update([], HAND)
        events = tracker.update([Card(Rank.NINE, Suit.CLUBS)], HAND)
        assert len(events) == 1
        assert events[0]['by_player'] is False
        assert state.unseen[Rank.NINE] == 3

    def test_pair_decrements_twice(self):
        tracker, state = make_tracker()
        tracker.update([], HAND)
        tracker.update([Card(Rank.NINE, Suit.CLUBS), Card(Rank.NINE, Suit.HEARTS)], HAND)
        assert state.unseen[Rank.NINE] == 2

    def test_covered_trick_is_not_replayed(self):
        """A new trick covering the old one must only report the new cards."""
        tracker, state = make_tracker()
        tracker.update([], HAND)
        tracker.update([Card(Rank.NINE, Suit.CLUBS)], HAND)
        events = tracker.update([Card(Rank.QUEEN, Suit.SPADES)], HAND)
        assert [e['cards'] for e in events] == [[Card(Rank.QUEEN, Suit.SPADES)]]
        assert state.unseen[Rank.NINE] == 3
        assert state.unseen[Rank.QUEEN] == 3

    def test_flickering_recognition_counts_once(self):
        """A card missing one frame and back the next is not a new play."""
        tracker, state = make_tracker()
        tracker.update([], HAND)
        trick = [Card(Rank.NINE, Suit.CLUBS)]
        tracker.update(trick, HAND)
        tracker.update([], HAND)
        events = tracker.update(trick, HAND)
        assert events == []
        assert state.unseen[Rank.NINE] == 3

    def test_second_joker_is_a_new_play(self):
        tracker, state = make_tracker()
        tracker.update([], HAND)
        tracker.update([Card(Rank.JOKER)], HAND)
        events = tracker.update([Card(Rank.JOKER)], HAND)
        assert len(events) == 1
        assert state.unseen[Rank.JOKER] == 0

    def test_impossible_play_raises(self):
        tracker, state = make_tracker(NINE=0)
        tracker.update([], HAND)
        with pytest.raises(ValueError, match='NINE'):
            tracker.update([Card(Rank.NINE, Suit.CLUBS)], HAND)


class TestOwnPlays:
    def test_own_play_does_not_touch_unseen(self):
        tracker, state = make_tracker()
        tracker.update([], HAND)
        played = [Card(Rank.FIVE, Suit.SPADES)]
        remaining = [Card(Rank.KING, Suit.HEARTS)]
        events = tracker.update(played, remaining)
        assert len(events) == 1
        assert events[0]['by_player'] is True
        assert state.unseen[Rank.FIVE] == 4

    def test_same_rank_from_opponent_still_counts(self):
        """Opponent plays a five while we hold a different five."""
        tracker, state = make_tracker()
        tracker.update([], HAND)
        events = tracker.update([Card(Rank.FIVE, Suit.HEARTS)], HAND)
        assert events[0]['by_player'] is False
        assert state.unseen[Rank.FIVE] == 3
