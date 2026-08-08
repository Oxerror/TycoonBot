import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from GameLogic.Card import Card, Rank, Suit
from GameLogic.Recommender import recommend
from GameLogic.Rules import PASS, set_rank


def c(rank, suit=None):
    return Card(rank, suit)


class TestFinishing:
    def test_emptying_the_hand_always_wins(self):
        hand = [c(Rank.TWO, Suit.HEARTS), c(Rank.TWO, Suit.SPADES)]
        trick = [c(Rank.FIVE, Suit.CLUBS), c(Rank.FIVE, Suit.HEARTS)]
        move = recommend(hand, trick)
        assert len(move) == 2

    def test_finishing_with_joker_is_fine(self):
        hand = [c(Rank.JOKER)]
        move = recommend(hand, [c(Rank.TWO, Suit.HEARTS)])
        assert set_rank(move) == Rank.JOKER


class TestLeading:
    def test_leads_weakest_set(self):
        hand = [c(Rank.THREE, Suit.CLUBS), c(Rank.KING, Suit.HEARTS),
                c(Rank.TWO, Suit.SPADES)]
        move = recommend(hand)
        assert set_rank(move) == Rank.THREE

    def test_prefers_whole_pair_over_breaking_it(self):
        hand = [c(Rank.FOUR, Suit.CLUBS), c(Rank.FOUR, Suit.HEARTS),
                c(Rank.KING, Suit.SPADES)]
        move = recommend(hand)
        assert set_rank(move) == Rank.FOUR
        assert len(move) == 2

    def test_does_not_lead_with_power_cards(self):
        hand = [c(Rank.FIVE, Suit.CLUBS), c(Rank.JOKER), c(Rank.WONDER)]
        move = recommend(hand)
        assert set_rank(move) == Rank.FIVE


class TestFollowing:
    def test_wins_with_cheapest_sufficient_card(self):
        hand = [c(Rank.QUEEN, Suit.CLUBS), c(Rank.TWO, Suit.HEARTS),
                c(Rank.FIVE, Suit.CLUBS)]
        move = recommend(hand, [c(Rank.TEN, Suit.HEARTS)])
        assert set_rank(move) == Rank.QUEEN

    def test_passes_instead_of_wasting_joker(self):
        hand = [c(Rank.JOKER), c(Rank.FIVE, Suit.CLUBS),
                c(Rank.FIVE, Suit.HEARTS), c(Rank.SIX, Suit.CLUBS),
                c(Rank.SIX, Suit.HEARTS)]
        move = recommend(hand, [c(Rank.TWO, Suit.HEARTS)])
        assert move == PASS

    def test_spends_joker_near_the_end(self):
        hand = [c(Rank.JOKER), c(Rank.FIVE, Suit.CLUBS),
                c(Rank.SIX, Suit.CLUBS)]
        move = recommend(hand, [c(Rank.TWO, Suit.HEARTS)])
        assert set_rank(move) == Rank.JOKER

    def test_passes_when_nothing_wins(self):
        hand = [c(Rank.FIVE, Suit.CLUBS), c(Rank.SIX, Suit.CLUBS)]
        move = recommend(hand, [c(Rank.TWO, Suit.HEARTS),
                                c(Rank.TWO, Suit.SPADES)])
        assert move == PASS

    def test_revolution_changes_the_pick(self):
        hand = [c(Rank.FIVE, Suit.CLUBS), c(Rank.KING, Suit.HEARTS)]
        trick = [c(Rank.TEN, Suit.HEARTS)]
        assert set_rank(recommend(hand, trick)) == Rank.KING
        assert set_rank(recommend(hand, trick, revolution=True)) == Rank.FIVE
