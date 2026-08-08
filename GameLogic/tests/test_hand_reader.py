import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from GameLogic.Card import Rank, Suit
from GameLogic.HandReader import detections_to_cards


def det(name, x, y, w=40, h=50, confidence=0.9):
    return {'name': name, 'confidence': confidence, 'location': (x, y, w, h)}


def as_tuples(cards):
    """(rank, suit) pairs — Card.__eq__ compares rank only, so tests must check suits explicitly."""
    return [(c.rank, c.suit) for c in cards]


class TestPairing:
    def test_rank_paired_with_suit_below(self):
        detections = [
            det('King', 100, 10, w=90, h=95),
            det('Heart', 120, 130),
        ]
        assert as_tuples(detections_to_cards(detections)) == [(Rank.KING, Suit.HEARTS)]

    def test_suit_above_rank_is_not_paired(self):
        detections = [
            det('King', 100, 200, w=90, h=95),
            det('Heart', 120, 10),
        ]
        assert detections_to_cards(detections) == []

    def test_horizontally_distant_suit_is_not_paired(self):
        detections = [
            det('King', 100, 10, w=90, h=95),
            det('Heart', 500, 130),
        ]
        assert detections_to_cards(detections) == []

    def test_nearest_suit_wins(self):
        detections = [
            det('King', 100, 10, w=90, h=95),
            det('Heart', 120, 130),
            det('Spade', 150, 190),
        ]
        assert as_tuples(detections_to_cards(detections)) == [(Rank.KING, Suit.HEARTS)]

    def test_confident_suit_preferred_over_closer_noise(self):
        """Card artwork can produce close low-confidence suit matches."""
        detections = [
            det('9', 100, 10, w=90, h=95),
            det('Heart', 110, 120, confidence=0.77),
            det('Cross', 130, 160, confidence=0.94),
        ]
        assert as_tuples(detections_to_cards(detections)) == [(Rank.NINE, Suit.CLUBS)]

    def test_each_suit_used_only_once(self):
        detections = [
            det('5', 100, 10, w=90, h=95),
            det('6', 200, 10, w=90, h=95),
            det('Heart', 145, 130),
        ]
        assert as_tuples(detections_to_cards(detections)) == [(Rank.FIVE, Suit.HEARTS)]

    def test_rank_without_suit_is_skipped(self):
        detections = [det('King', 100, 10, w=90, h=95)]
        assert detections_to_cards(detections) == []

    def test_neighbor_suit_is_not_stolen(self):
        """A card whose own suit scored just below the confidence cutoff
        must not grab the neighbor card's confident suit — that sets off
        a chain where every card takes its right neighbor's suit."""
        detections = [
            det('Jack', 100, 10, w=100, h=110),
            det('Cross', 122, 120, confidence=0.84),
            det('Queen', 220, 10, w=100, h=110),
            det('Cross', 248, 120, confidence=0.92),
        ]
        assert as_tuples(detections_to_cards(detections)) == [
            (Rank.JACK, Suit.CLUBS),
            (Rank.QUEEN, Suit.CLUBS),
        ]


class TestSpecialCards:
    def test_joker_needs_no_suit(self):
        cards = detections_to_cards([det('Joker', 100, 10)])
        assert as_tuples(cards) == [(Rank.JOKER, None)]

    def test_wonder_needs_no_suit(self):
        cards = detections_to_cards([det('Wonder', 100, 10)])
        assert as_tuples(cards) == [(Rank.WONDER, None)]


class TestOrdering:
    def test_cards_sorted_left_to_right(self):
        detections = [
            det('King', 400, 10, w=90, h=95),
            det('Spade', 420, 130),
            det('3', 100, 10, w=90, h=95),
            det('Heart', 120, 130),
        ]
        assert as_tuples(detections_to_cards(detections)) == [
            (Rank.THREE, Suit.HEARTS),
            (Rank.KING, Suit.SPADES),
        ]

    def test_empty_detections(self):
        assert detections_to_cards([]) == []
