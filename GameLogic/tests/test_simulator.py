import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from GameLogic.Card import Rank
from GameLogic.Simulator import (build_deck, deal, first_leader, play_round,
                                 random_policy, recommender_policy)


class TestDealing:
    def test_deck_is_complete(self):
        deck = build_deck()
        assert len(deck) == 56
        counts = Counter(card.rank for card in deck)
        assert counts[Rank.JOKER] == 2
        assert counts[Rank.WONDER] == 2
        assert counts[Rank.ACE] == 4

    def test_deal_splits_evenly_and_loses_nothing(self):
        hands = deal(random.Random(7))
        assert [len(h) for h in hands] == [14, 14, 14, 14]
        dealt = Counter((c.rank, c.suit) for hand in hands for c in hand)
        full = Counter((c.rank, c.suit) for c in build_deck())
        assert dealt == full

    def test_first_leader_holds_the_three_of_diamonds(self):
        hands = deal(random.Random(7))
        leader = first_leader(hands)
        assert any(card.rank == Rank.THREE and card.suit is not None
                   and card.suit.name == 'DIAMONDS'
                   for card in hands[leader])


class TestFullRounds:
    def test_random_round_completes(self):
        rng = random.Random(42)
        hands = deal(rng)
        policies = [random_policy(rng) for _ in range(4)]
        ranking = play_round(hands, first_leader(hands), policies)
        assert sorted(ranking) == [0, 1, 2, 3]

    def test_many_random_rounds_complete(self):
        rng = random.Random(1)
        for _ in range(25):
            hands = deal(rng)
            policies = [random_policy(rng) for _ in range(4)]
            ranking = play_round(hands, first_leader(hands), policies)
            assert sorted(ranking) == [0, 1, 2, 3]

    def test_events_report_a_coherent_round(self):
        rng = random.Random(3)
        hands = deal(rng)
        events = []
        play_round(hands, first_leader(hands),
                   [random_policy(rng) for _ in range(4)], events.append)
        played = Counter()
        for event in events:
            if event[0] == 'play':
                played.update((c.rank, c.suit) for c in event[2])
        # No card hits the table more often than the deck holds it...
        full = Counter((c.rank, c.suit) for c in build_deck())
        assert not played - full
        # ...and the three finishers shed all 14 cards each; only the
        # Beggar keeps a remainder when the round ends.
        assert sum(played.values()) >= 42
        assert [e for e in events if e[0] == 'round_over']


class TestRecommenderStrength:
    def test_recommender_outranks_random_players(self):
        """Seat 0 runs the real recommender against three random
        players; over seeded rounds it must finish first far more
        often than the 25% a fair seat would."""
        rng = random.Random(2026)
        wins = 0
        rounds = 40
        for _ in range(rounds):
            hands = deal(rng)
            policies = [recommender_policy] + [random_policy(rng)
                                               for _ in range(3)]
            ranking = play_round(hands, first_leader(hands), policies)
            if ranking[0] == 0:
                wins += 1
        assert wins >= rounds // 2, f"recommender won only {wins}/{rounds}"
