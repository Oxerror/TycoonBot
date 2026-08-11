import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from GameLogic.Card import Card, Rank, Suit
from GameLogic.Simulator import (best_cards, build_deck, deal, exchange_cards,
                                 first_leader, play_game, play_round,
                                 random_policy, recommender_policy,
                                 weakest_cards)


def c(rank, suit=None):
    return Card(rank, suit)


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


class TestExchangeChoices:
    def test_best_cards_are_the_strongest_but_never_the_wonder(self):
        hand = [c(Rank.WONDER), c(Rank.JOKER), c(Rank.TWO, Suit.HEARTS),
                c(Rank.ACE, Suit.CLUBS), c(Rank.FOUR, Suit.SPADES)]
        best = best_cards(hand, 2)
        assert [card.rank for card in best] == [Rank.JOKER, Rank.TWO]

    def test_weakest_cards_shed_the_low_ranks(self):
        hand = [c(Rank.WONDER), c(Rank.THREE, Suit.HEARTS),
                c(Rank.KING, Suit.CLUBS), c(Rank.FOUR, Suit.SPADES)]
        weakest = weakest_cards(hand, 2)
        assert [card.rank for card in weakest] == [Rank.THREE, Rank.FOUR]


class TestExchange:
    @staticmethod
    def hands():
        return [
            # Tycoon: 3♥ and 4♣ are the weakest, given back to the Beggar.
            [c(Rank.KING, Suit.CLUBS), c(Rank.THREE, Suit.HEARTS),
             c(Rank.FOUR, Suit.CLUBS), c(Rank.TEN, Suit.SPADES)],
            # Rich: gives back the 5♦.
            [c(Rank.FIVE, Suit.DIAMONDS), c(Rank.JOKER),
             c(Rank.NINE, Suit.HEARTS)],
            # Poor: the Joker is its best card despite the Wonder.
            [c(Rank.WONDER), c(Rank.JOKER), c(Rank.SIX, Suit.CLUBS)],
            # Beggar: 2♠ and Ace are the tribute; the Wonder stays.
            [c(Rank.TWO, Suit.SPADES), c(Rank.WONDER),
             c(Rank.ACE, Suit.DIAMONDS), c(Rank.SEVEN, Suit.HEARTS)],
        ]

    def test_tributes_flow_up_and_returns_flow_down(self):
        hands = self.hands()
        events = []
        exchange_cards(hands, ranking=[0, 1, 2, 3], on_event=events.append)

        tycoon_ranks = {card.rank for card in hands[0]}
        assert {Rank.TWO, Rank.ACE} <= tycoon_ranks
        assert Rank.THREE not in tycoon_ranks and Rank.FOUR not in tycoon_ranks
        beggar_ranks = {card.rank for card in hands[3]}
        assert {Rank.THREE, Rank.FOUR, Rank.WONDER} <= beggar_ranks

        rich_ranks = [card.rank for card in hands[1]]
        assert rich_ranks.count(Rank.JOKER) == 2
        assert Rank.FIVE not in rich_ranks
        poor_ranks = {card.rank for card in hands[2]}
        assert poor_ranks == {Rank.WONDER, Rank.FIVE, Rank.SIX}

        assert [event[0] for event in events] == ['exchange', 'exchange']
        assert events[0][1:3] == (3, 0) and events[1][1:3] == (2, 1)

    def test_hand_sizes_survive_the_exchange(self):
        hands = self.hands()
        sizes = [len(hand) for hand in hands]
        exchange_cards(hands, ranking=[0, 1, 2, 3])
        assert [len(hand) for hand in hands] == sizes

    def test_ranking_indexes_seats_not_places(self):
        hands = self.hands()
        # Seat 3 was Tycoon, seat 0 Beggar: tribute K♣ + 10♠ flows
        # 0 -> 3; the weakest two of the merged hand (7♥, 10♠) return.
        exchange_cards(hands, ranking=[3, 2, 1, 0])
        seat3_ranks = {card.rank for card in hands[3]}
        assert seat3_ranks == {Rank.TWO, Rank.WONDER, Rank.ACE, Rank.KING}
        seat0_ranks = {card.rank for card in hands[0]}
        assert seat0_ranks == {Rank.THREE, Rank.FOUR, Rank.SEVEN, Rank.TEN}

    def test_exchange_hook_overrides_the_default_return(self):
        class Chooser:
            def __call__(self, obs):
                raise AssertionError("never plays in this test")

            def exchange(self, hand, count):
                # Return the 3♠ even though an equal-rank 3♦ sits
                # earlier in the hand — suits must be honored — plus
                # the King to make up the count.
                spade_three = next(card for card in hand
                                   if card.rank == Rank.THREE
                                   and card.suit == Suit.SPADES)
                king = next(card for card in hand
                            if card.rank == Rank.KING)
                return [spade_three, king][:count]

        hands = self.hands()
        hands[0] = [c(Rank.THREE, Suit.DIAMONDS), c(Rank.THREE, Suit.SPADES),
                    c(Rank.KING, Suit.CLUBS)]
        policies = [Chooser(), None, None, None]
        exchange_cards(hands, ranking=[0, 1, 2, 3], policies=policies)
        suits = [card.suit for card in hands[0]
                 if card.rank == Rank.THREE]
        assert suits == [Suit.DIAMONDS]
        assert any(card.rank == Rank.THREE and card.suit == Suit.SPADES
                   for card in hands[3])


class TestGames:
    def test_game_plays_rounds_and_exchanges_between_them(self):
        rng = random.Random(11)
        policies = [random_policy(rng) for _ in range(4)]
        events = []
        rankings = play_game(policies, rounds=3, rng=rng,
                             on_event=events.append)
        assert len(rankings) == 3
        assert all(sorted(r) == [0, 1, 2, 3] for r in rankings)
        kinds = Counter(event[0] for event in events)
        assert kinds['round_start'] == 3
        assert kinds['exchange'] == 4      # two tributes per later round
        assert kinds['round_over'] == 3

    def test_beggar_leads_the_later_rounds(self):
        rng = random.Random(5)
        policies = [random_policy(rng) for _ in range(4)]
        events = []
        rankings = play_game(policies, rounds=4, rng=rng,
                             on_event=events.append)
        starts = [event for event in events if event[0] == 'round_start']
        for previous, start in zip(rankings, starts[1:]):
            assert start[2] == previous[-1]

    def test_games_are_seed_reproducible(self):
        def run():
            rng = random.Random(23)
            policies = [random_policy(rng) for _ in range(4)]
            return play_game(policies, rounds=3, rng=rng)
        assert run() == run()


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
