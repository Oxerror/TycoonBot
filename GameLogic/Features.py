"""Encode a (situation, candidate move) pair as a feature vector.

The learned evaluator scores candidate moves the same way the rollout
search does — "how well do I finish if I play this?" — so both see
the same information: the own hand, the unseen counts, everyone's
hand sizes and pass states, the trick, and what the move does to the
hand. All values are scaled to roughly [0, 1].

The encoding is deliberately small and hand-shaped; it is the input
contract of PolicyNet, so FEATURE_COUNT changes must retrain models.
"""

from GameLogic.Card import Rank
from GameLogic.Rules import (causes_revolution, contains_wonder,
                             is_unbeatable, move_strength,
                             wins_trick_immediately)

RANKS = list(Rank)
# 4 rank histograms + 17 scalars + match context: a no-roles flag,
# a role one-hot for the own seat and each opponent, the round index.
FEATURE_COUNT = 4 * len(RANKS) + 17 + 1 + 4 * 4 + 1


def _rank_counts(cards):
    counts = [0.0] * len(RANKS)
    for card in cards:
        counts[card.rank.value - 1] += 0.25
    return counts


def encode(obs, move):
    """
    Feature vector for playing `move` (or PASS: the empty tuple) in
    the situation `obs` (a Simulator Observation).
    """
    hand = list(obs.hand)
    remaining = list(hand)
    for card in move:
        remaining.remove(card)

    others = [(obs.seat + offset) % len(obs.counts)
              for offset in (1, 2, 3)]

    features = []
    features += _rank_counts(hand)
    features += _rank_counts(remaining)
    features += _rank_counts(move)
    features += [obs.unseen[rank] / 4 for rank in RANKS]

    features.append(len(hand) / 14)
    features += [obs.counts[seat] / 14 for seat in others]
    features += [1.0 if seat in obs.passed else 0.0 for seat in others]

    # Match context: the exchange stacks the Tycoon's hand and strips
    # the Beggar's, so who holds which role — and whether roles exist
    # at all yet — changes what the same cards are worth.
    features.append(1.0 if obs.roles is None else 0.0)
    for seat in [obs.seat] + others:
        one_hot = [0.0] * 4
        if obs.roles is not None:
            one_hot[obs.roles[seat]] = 1.0
        features += one_hot
    features.append(obs.round_index / 3)

    features.append(1.0 if obs.trick else 0.0)
    features.append(len(obs.trick) / 4)
    features.append(move_strength(obs.trick, obs.revolution) / 200
                    if obs.trick else 0.0)
    features.append(1.0 if obs.revolution else 0.0)

    is_pass = not move
    features.append(1.0 if is_pass else 0.0)
    if is_pass:
        features += [0.0] * 5
    else:
        features.append(move_strength(move, obs.revolution) / 200)
        features.append(1.0 if contains_wonder(move) else 0.0)
        features.append(1.0 if causes_revolution(move) else 0.0)
        features.append(1.0 if wins_trick_immediately(move, obs.trick) else 0.0)
        features.append(1.0 if is_unbeatable(move, obs.unseen,
                                             obs.revolution) else 0.0)

    assert len(features) == FEATURE_COUNT
    return features
