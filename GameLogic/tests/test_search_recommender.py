import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from GameLogic.Card import Card, Rank, Suit
from GameLogic.Rules import PASS, set_rank
from GameLogic.SearchRecommender import SearchPolicy
from GameLogic.Simulator import (Observation, deal, first_leader, play_round,
                                 recommender_policy)


def c(rank, suit=None):
    return Card(rank, suit)


def unseen(**counts):
    result = {rank: 0 for rank in Rank}
    for name, count in counts.items():
        result[Rank[name]] = count
    return result


def observe(hand, trick=(), revolution=False, counts_others=(5, 5, 5),
            **unseen_counts):
    """An Observation for seat 0 with the given unseen ranks."""
    return Observation(seat=0, hand=tuple(hand), trick=tuple(trick),
                       revolution=revolution,
                       unseen=unseen(**unseen_counts),
                       counts=(len(hand),) + tuple(counts_others),
                       passed=frozenset(),
                       last_player=3 if trick else None)


class TestShortcuts:
    def test_plays_the_finisher_without_sampling(self):
        policy = SearchPolicy(samples=0, rng=random.Random(1))
        obs = observe([c(Rank.FIVE, Suit.CLUBS)], trick=[c(Rank.FOUR, Suit.HEARTS)],
                      KING=4, ACE=4, TWO=4, JOKER=2, WONDER=2)
        move = policy(obs)
        assert set_rank(move) == Rank.FIVE

    def test_plays_the_guaranteed_win_without_sampling(self):
        policy = SearchPolicy(samples=0, rng=random.Random(1))
        obs = observe([c(Rank.TWO, Suit.HEARTS), c(Rank.FIVE, Suit.CLUBS)],
                      KING=2, QUEEN=2)
        move = policy(obs)
        assert set_rank(move) == Rank.TWO

    def test_passes_when_nothing_beats_the_trick(self):
        policy = SearchPolicy(samples=0, rng=random.Random(1))
        obs = observe([c(Rank.FIVE, Suit.CLUBS), c(Rank.SIX, Suit.CLUBS)],
                      trick=[c(Rank.TWO, Suit.HEARTS)], KING=4, JOKER=2)
        assert policy(obs) == PASS


class TestSearchedRounds:
    def test_search_policy_survives_full_rounds(self):
        rng = random.Random(11)
        for _ in range(3):
            hands = deal(rng)
            policies = [SearchPolicy(samples=2, rng=random.Random(rng.random()))
                        for _ in range(4)]
            ranking = play_round(hands, first_leader(hands), policies)
            assert sorted(ranking) == [0, 1, 2, 3]

    @pytest.mark.slow
    def test_search_outranks_the_heuristic(self):
        """Seat 0 searches while the other three run the heuristic;
        over seeded rounds the searcher must finish Tycoon clearly
        more often than the 25% a fair seat would."""
        rng = random.Random(2026)
        wins = 0
        rounds = 30
        for _ in range(rounds):
            hands = deal(rng)
            policies = ([SearchPolicy(samples=12, rng=random.Random(rng.random()))]
                        + [recommender_policy] * 3)
            ranking = play_round(hands, first_leader(hands), policies)
            if ranking[0] == 0:
                wins += 1
        assert wins >= 12, f"search won only {wins}/{rounds}"
