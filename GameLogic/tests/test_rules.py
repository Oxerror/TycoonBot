import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from GameLogic.Card import Card, Rank, Suit
from GameLogic.Rules import (causes_revolution, effective_strength,
                             is_spade_three_counter, is_unbeatable,
                             legal_moves, move_strength, set_rank,
                             wins_trick_immediately)


def c(rank, suit=None):
    return Card(rank, suit)


def unseen(**counts):
    """Unseen-count dict: every rank 0 except the given overrides."""
    result = {rank: 0 for rank in Rank}
    for name, count in counts.items():
        result[Rank[name]] = count
    return result


class TestStrength:
    def test_normal_order(self):
        assert effective_strength(Rank.THREE) < effective_strength(Rank.TWO)
        assert effective_strength(Rank.ACE) < effective_strength(Rank.TWO)
        assert effective_strength(Rank.KING) < effective_strength(Rank.ACE)

    def test_joker_beats_two(self):
        assert effective_strength(Rank.TWO) < effective_strength(Rank.JOKER)

    def test_revolution_flips_normal_ranks(self):
        assert (effective_strength(Rank.THREE, revolution=True)
                > effective_strength(Rank.TWO, revolution=True))

    def test_joker_unaffected_by_revolution(self):
        assert (effective_strength(Rank.JOKER, revolution=True)
                > effective_strength(Rank.THREE, revolution=True))

    def test_wonder_above_everything(self):
        assert effective_strength(Rank.WONDER) > effective_strength(Rank.JOKER)


class TestSetRank:
    def test_pair_rank(self):
        assert set_rank([c(Rank.KING, Suit.HEARTS), c(Rank.KING, Suit.SPADES)]) == Rank.KING

    def test_joker_assumes_set_rank(self):
        assert set_rank([c(Rank.KING, Suit.HEARTS), c(Rank.JOKER)]) == Rank.KING

    def test_joker_only_set(self):
        assert set_rank([c(Rank.JOKER), c(Rank.JOKER)]) == Rank.JOKER

    def test_mixed_ranks_rejected(self):
        with pytest.raises(ValueError):
            set_rank([c(Rank.KING, Suit.HEARTS), c(Rank.QUEEN, Suit.SPADES)])


class TestLeading:
    def test_every_set_size_available(self):
        hand = [c(Rank.KING, Suit.HEARTS), c(Rank.KING, Suit.SPADES),
                c(Rank.FIVE, Suit.CLUBS)]
        sizes = {(set_rank(m), len(m)) for m in legal_moves(hand)}
        assert (Rank.KING, 1) in sizes
        assert (Rank.KING, 2) in sizes
        assert (Rank.FIVE, 1) in sizes

    def test_joker_extends_sets(self):
        hand = [c(Rank.KING, Suit.HEARTS), c(Rank.JOKER)]
        sizes = {(set_rank(m), len(m)) for m in legal_moves(hand)}
        assert (Rank.KING, 2) in sizes  # King + Joker as a pair
        assert (Rank.JOKER, 1) in sizes

    def test_wonder_playable_as_lead(self):
        hand = [c(Rank.WONDER), c(Rank.FIVE, Suit.CLUBS)]
        assert any(set_rank(m) == Rank.WONDER for m in legal_moves(hand))


class TestFollowing:
    TRICK = [c(Rank.TEN, Suit.HEARTS)]

    def test_higher_single_allowed(self):
        moves = legal_moves([c(Rank.KING, Suit.HEARTS)], self.TRICK)
        assert len(moves) == 1

    def test_lower_single_not_allowed(self):
        assert legal_moves([c(Rank.FIVE, Suit.CLUBS)], self.TRICK) == []

    def test_equal_rank_not_allowed(self):
        assert legal_moves([c(Rank.TEN, Suit.SPADES)], self.TRICK) == []

    def test_size_must_match(self):
        hand = [c(Rank.KING, Suit.HEARTS), c(Rank.KING, Suit.SPADES)]
        pair_trick = [c(Rank.TEN, Suit.HEARTS), c(Rank.TEN, Suit.SPADES)]
        moves = legal_moves(hand, pair_trick)
        assert all(len(m) == 2 for m in moves)
        single_answers = legal_moves(hand, self.TRICK)
        assert all(len(m) == 1 for m in single_answers)

    def test_joker_beats_two(self):
        moves = legal_moves([c(Rank.JOKER)], [c(Rank.TWO, Suit.HEARTS)])
        assert len(moves) == 1

    def test_revolution_flips_legality(self):
        hand = [c(Rank.FIVE, Suit.CLUBS)]
        assert legal_moves(hand, self.TRICK) == []
        assert len(legal_moves(hand, self.TRICK, revolution=True)) == 1

    def test_joker_still_wins_under_revolution(self):
        moves = legal_moves([c(Rank.JOKER)], [c(Rank.THREE, Suit.HEARTS)],
                            revolution=True)
        assert len(moves) == 1


class TestSpadeThreeReversal:
    def test_spade_three_beats_single_joker(self):
        moves = legal_moves([c(Rank.THREE, Suit.SPADES)], [c(Rank.JOKER)])
        assert len(moves) == 1

    def test_other_threes_do_not(self):
        assert legal_moves([c(Rank.THREE, Suit.DIAMONDS)], [c(Rank.JOKER)]) == []

    def test_only_single_vs_single(self):
        pair_hand = [c(Rank.THREE, Suit.SPADES), c(Rank.THREE, Suit.HEARTS)]
        joker_pair = [c(Rank.JOKER), c(Rank.JOKER)]
        assert all(not is_spade_three_counter(m, joker_pair)
                   for m in legal_moves(pair_hand, joker_pair))

    def test_counter_wins_trick_immediately(self):
        move = (c(Rank.THREE, Suit.SPADES),)
        assert wins_trick_immediately(move, [c(Rank.JOKER)])
        assert not wins_trick_immediately(move, [c(Rank.TWO, Suit.HEARTS)])


class TestWonder:
    def test_wonder_beats_any_trick_size(self):
        hand = [c(Rank.WONDER)]
        pair_trick = [c(Rank.TWO, Suit.HEARTS), c(Rank.TWO, Suit.SPADES)]
        assert len(legal_moves(hand, pair_trick)) == 1

    def test_wonder_wins_immediately(self):
        assert wins_trick_immediately((c(Rank.WONDER),))


class TestEightStop:
    def test_eight_set_wins_immediately(self):
        assert wins_trick_immediately((c(Rank.EIGHT, Suit.CLUBS),))
        assert wins_trick_immediately(
            (c(Rank.EIGHT, Suit.CLUBS), c(Rank.JOKER)))

    def test_other_sets_do_not(self):
        assert not wins_trick_immediately((c(Rank.NINE, Suit.CLUBS),))


class TestRevolutionTrigger:
    def test_four_of_a_kind(self):
        move = tuple(c(Rank.NINE, s) for s in Suit)
        assert causes_revolution(move)

    def test_four_with_joker(self):
        move = (c(Rank.NINE, Suit.CLUBS), c(Rank.NINE, Suit.HEARTS),
                c(Rank.NINE, Suit.SPADES), c(Rank.JOKER))
        assert causes_revolution(move)

    def test_smaller_sets_do_not(self):
        assert not causes_revolution((c(Rank.NINE, Suit.CLUBS),
                                      c(Rank.NINE, Suit.HEARTS)))


class TestUnbeatable:
    def test_ace_unbeatable_once_twos_and_jokers_are_gone(self):
        move = (c(Rank.ACE, Suit.HEARTS),)
        assert is_unbeatable(move, unseen(KING=3, QUEEN=4))
        assert not is_unbeatable(move, unseen(TWO=1))
        assert not is_unbeatable(move, unseen(JOKER=1))

    def test_unseen_wonder_beats_everything(self):
        move = (c(Rank.TWO, Suit.HEARTS),)
        assert not is_unbeatable(move, unseen(WONDER=1))

    def test_pair_needs_a_full_unseen_pair_to_beat(self):
        move = (c(Rank.KING, Suit.HEARTS), c(Rank.KING, Suit.SPADES))
        assert is_unbeatable(move, unseen(ACE=1, TWO=1))
        assert not is_unbeatable(move, unseen(ACE=2))

    def test_unseen_joker_completes_a_beating_pair(self):
        move = (c(Rank.KING, Suit.HEARTS), c(Rank.KING, Suit.SPADES))
        assert not is_unbeatable(move, unseen(ACE=1, JOKER=1))

    def test_jokers_alone_form_the_strongest_set(self):
        move = (c(Rank.TWO, Suit.HEARTS), c(Rank.TWO, Suit.SPADES))
        assert not is_unbeatable(move, unseen(JOKER=2))
        # A single unseen joker cannot pair up against a pair of 2s.
        assert is_unbeatable(move, unseen(JOKER=1))

    def test_lone_joker_fears_the_spade_three(self):
        move = (c(Rank.JOKER),)
        assert not is_unbeatable(move, unseen(THREE=1))
        assert is_unbeatable(move, unseen(FOUR=4, TWO=4))

    def test_joker_pair_does_not_fear_threes(self):
        move = (c(Rank.JOKER), c(Rank.JOKER))
        assert is_unbeatable(move, unseen(THREE=4, TWO=4))
        assert not is_unbeatable(move, unseen(WONDER=1))

    def test_immediate_winners_are_always_unbeatable(self):
        assert is_unbeatable((c(Rank.EIGHT, Suit.CLUBS),), unseen(WONDER=2, JOKER=2))
        assert is_unbeatable((c(Rank.WONDER),), unseen(WONDER=1, JOKER=2))

    def test_revolution_flips_the_judgement(self):
        move = (c(Rank.FOUR, Suit.HEARTS),)
        threat = unseen(THREE=2)
        assert is_unbeatable(move, threat)
        assert not is_unbeatable(move, threat, revolution=True)


class TestSuitlessPlaceholders:
    def test_placeholder_cards_are_playable(self):
        """Cards recovered from the bar have no suit but a known rank."""
        moves = legal_moves([c(Rank.KING)], [c(Rank.TEN, Suit.HEARTS)])
        assert len(moves) == 1
