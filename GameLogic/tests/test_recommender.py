import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from GameLogic.Card import Card, Rank, Suit
from GameLogic.Recommender import recommend
from GameLogic.Rules import PASS, set_rank


def c(rank, suit=None):
    return Card(rank, suit)


def unseen(**counts):
    """Unseen-count dict: every rank 0 except the given overrides."""
    result = {rank: 0 for rank in Rank}
    for name, count in counts.items():
        result[Rank[name]] = count
    return result


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


class TestUnseenAware:
    """With GameState's unseen counts the recommender recognizes
    guaranteed finishes: a chain of unbeatable sets ending in any
    final set wins the round on the spot."""

    # A joker the v1 heuristics would hoard, plus two pairs.
    JOKER_HAND = [c(Rank.JOKER), c(Rank.FIVE, Suit.CLUBS),
                  c(Rank.FIVE, Suit.HEARTS), c(Rank.SIX, Suit.CLUBS),
                  c(Rank.SIX, Suit.HEARTS)]
    TWO_TRICK = [c(Rank.TWO, Suit.HEARTS)]

    def test_spends_the_joker_when_the_round_is_mathematically_won(self):
        # Joker unbeatable (no 3s, no Wonder), pair of 5s unbeatable
        # (no unseen pair above it), pair of 6s is the free last set.
        counts = unseen(KING=1, QUEEN=1)
        move = recommend(self.JOKER_HAND, self.TWO_TRICK, unseen=counts)
        assert set_rank(move) == Rank.JOKER

    def test_still_passes_while_an_unseen_three_threatens_the_joker(self):
        # Any unseen 3 might be the 3 of Spades, so the joker is not
        # a sure trick-winner and the v1 caution stands.
        counts = unseen(THREE=1)
        assert recommend(self.JOKER_HAND, self.TWO_TRICK, unseen=counts) == PASS

    def test_still_passes_while_the_wonder_is_out(self):
        counts = unseen(WONDER=1)
        assert recommend(self.JOKER_HAND, self.TWO_TRICK, unseen=counts) == PASS

    def test_leads_the_unbeatable_two_to_lock_the_round(self):
        # v1 leads the weakest card; knowing the 2 cannot be topped,
        # play it first and finish with the 5 uncontested.
        hand = [c(Rank.FIVE, Suit.CLUBS), c(Rank.TWO, Suit.HEARTS)]
        assert set_rank(recommend(hand)) == Rank.FIVE
        counts = unseen(KING=2)
        assert set_rank(recommend(hand, unseen=counts)) == Rank.TWO

    def test_lead_out_respects_its_own_revolution(self):
        # The quad flips the order, which is exactly what makes the 3
        # unbeatable afterwards; the 4 is the free last set.
        hand = [c(Rank.NINE, s) for s in Suit]
        hand += [c(Rank.THREE, Suit.CLUBS), c(Rank.FOUR, Suit.HEARTS)]
        counts = unseen(TWO=2)
        move = recommend(hand, unseen=counts)
        assert set_rank(move) == Rank.NINE
        assert len(move) == 4

    def test_no_guarantee_without_unseen_counts(self):
        assert recommend(self.JOKER_HAND, self.TWO_TRICK) == PASS
