import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from GameLogic.Card import Card, Rank, Suit
from GameLogic.GameState import (DECK_SIZE, FULL_DECK, GameState,
                                 expected_hand_ranks, validate_start_hand)


def full_counts(**overrides):
    counts = dict(FULL_DECK)
    for name, value in overrides.items():
        counts[Rank[name]] = value
    return counts


class TestDeck:
    def test_deck_size(self):
        assert DECK_SIZE == 56

    def test_composition(self):
        assert FULL_DECK[Rank.JOKER] == 2
        assert FULL_DECK[Rank.WONDER] == 2
        for rank in Rank:
            if rank not in (Rank.JOKER, Rank.WONDER):
                assert FULL_DECK[rank] == 4


class TestConstruction:
    def test_from_status_bar(self):
        state = GameState.from_status_bar(full_counts())
        assert state.unseen[Rank.KING] == 4
        assert state.unseen[Rank.WONDER] == 2

    def test_missing_rank_rejected(self):
        counts = full_counts()
        del counts[Rank.KING]
        with pytest.raises(ValueError, match='KING'):
            GameState(counts)

    def test_total_unseen(self):
        assert GameState(full_counts()).total_unseen() == DECK_SIZE


class TestStartHandValidation:
    """At round start, deck - bar = the own hand."""

    def hand(self):
        return [Card(Rank.KING, Suit.HEARTS), Card(Rank.KING, Suit.SPADES),
                Card(Rank.THREE, Suit.CLUBS), Card(Rank.JOKER)]

    def bar_for(self, hand):
        counts = full_counts()
        for card in hand:
            counts[card.rank] -= 1
        return counts

    def test_perfect_reading_validates(self):
        hand = self.hand()
        missing, extra = validate_start_hand(hand, self.bar_for(hand))
        assert missing == {} and extra == {}

    def test_clipped_cards_reported_missing(self):
        hand = self.hand()
        bar = self.bar_for(hand)
        read = [c for c in hand if c.rank != Rank.JOKER]
        missing, extra = validate_start_hand(read, bar)
        assert missing == {Rank.JOKER: 1}
        assert extra == {}

    def test_impossible_card_reported_extra(self):
        hand = self.hand()
        bar = self.bar_for(hand)
        read = hand + [Card(Rank.ACE, Suit.HEARTS)]
        missing, extra = validate_start_hand(read, bar)
        assert extra == {Rank.ACE: 1}
        assert missing == {}

    def test_expected_hand_ranks(self):
        hand = self.hand()
        expected = expected_hand_ranks(self.bar_for(hand))
        assert expected[Rank.KING] == 2
        assert expected[Rank.THREE] == 1
        assert expected[Rank.JOKER] == 1
        assert expected[Rank.ACE] == 0


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
