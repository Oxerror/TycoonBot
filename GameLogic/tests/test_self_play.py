import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from GameLogic.Card import Card, Rank, Suit
from GameLogic.Features import FEATURE_COUNT, encode
from GameLogic.Rules import PASS
from GameLogic.SearchRecommender import SearchPolicy
from GameLogic.SelfPlay import generate_dataset
from GameLogic.Simulator import Observation


def c(rank, suit=None):
    return Card(rank, suit)


def observe(hand, trick=(), **overrides):
    unseen = {rank: 0 for rank in Rank}
    unseen[Rank.KING] = 2
    fields = dict(seat=0, hand=tuple(hand), trick=tuple(trick),
                  revolution=False, unseen=unseen, counts=(len(hand), 5, 5, 5),
                  passed=frozenset(), last_player=3 if trick else None)
    fields.update(overrides)
    return Observation(**fields)


class TestFeatures:
    HAND = [c(Rank.FIVE, Suit.CLUBS), c(Rank.FIVE, Suit.HEARTS),
            c(Rank.ACE, Suit.SPADES)]

    def test_vector_shape_and_range(self):
        obs = observe(self.HAND, trick=[c(Rank.FOUR, Suit.HEARTS)])
        vector = encode(obs, (self.HAND[0],))
        assert len(vector) == FEATURE_COUNT
        assert all(0.0 <= value <= 1.0 for value in vector)

    def test_pass_and_play_differ(self):
        obs = observe(self.HAND, trick=[c(Rank.FOUR, Suit.HEARTS)])
        assert encode(obs, PASS) != encode(obs, (self.HAND[0],))

    def test_encoding_is_deterministic(self):
        obs = observe(self.HAND)
        move = (self.HAND[0], self.HAND[1])
        assert encode(obs, move) == encode(obs, move)


class TestRecorder:
    def test_recorder_taps_sampled_decisions(self):
        taps = []
        policy = SearchPolicy(samples=2, rng=random.Random(3),
                              recorder=lambda *args: taps.append(args))
        hand = [c(Rank.FIVE, Suit.CLUBS), c(Rank.NINE, Suit.HEARTS),
                c(Rank.KING, Suit.SPADES)]
        obs = observe(hand, trick=[c(Rank.FOUR, Suit.HEARTS)])
        policy(obs)
        assert len(taps) == 1
        _, candidates, averages = taps[0]
        assert len(candidates) == len(averages)
        assert PASS in candidates
        assert all(0.0 <= place <= 3.0 for place in averages)

    def test_shortcut_decisions_are_not_recorded(self):
        taps = []
        policy = SearchPolicy(samples=2, rng=random.Random(3),
                              recorder=lambda *args: taps.append(args))
        obs = observe([c(Rank.FIVE, Suit.CLUBS)],
                      trick=[c(Rank.FOUR, Suit.HEARTS)])
        move = policy(obs)          # finisher: no sampling happened
        assert len(move) == 1
        assert taps == []


class TestDataset:
    def test_tiny_self_play_produces_samples(self):
        features, places = generate_dataset(rounds=1, samples=2, seed=5)
        assert len(features) == len(places)
        assert len(features) > 10
        assert all(len(vector) == FEATURE_COUNT for vector in features)
        assert all(0.0 <= place <= 3.0 for place in places)


@pytest.mark.slow
class TestTraining:
    def test_training_learns_and_policy_plays_legal(self):
        from GameLogic.PolicyNet import LearnedPolicy, train
        features, places = generate_dataset(rounds=2, samples=3, seed=11)
        losses = []
        model = train(features, places, epochs=12, device='cpu',
                      log=lambda e, loss: losses.append(loss))
        assert losses[-1] < losses[0]

        policy = LearnedPolicy(model)
        from GameLogic.Rules import legal_moves
        hand = [c(Rank.FIVE, Suit.CLUBS), c(Rank.NINE, Suit.HEARTS),
                c(Rank.KING, Suit.SPADES), c(Rank.KING, Suit.HEARTS)]
        obs = observe(hand, trick=[c(Rank.FOUR, Suit.HEARTS)])
        move = policy(obs)
        legal = legal_moves(hand, obs.trick, obs.revolution)
        assert move == PASS or move in legal
