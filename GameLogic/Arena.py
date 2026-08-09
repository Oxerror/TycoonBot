"""Pit recommender policies against each other over seeded rounds.

The measuring stick for every recommender idea: hand four policy
factories to `compare` and read off who finishes where. A factory is
a callable rng -> policy, so stochastic policies get a fresh seeded
generator every round and runs stay reproducible.

Run directly for a quick scoreboard:

    .venv\\Scripts\\python -m GameLogic.Arena
"""

import random
from collections import Counter

from GameLogic.Simulator import deal, first_leader, play_round

PLACE_NAMES = ('Tycoon', 'Rich', 'Poor', 'Beggar')


def compare(factories, rounds=100, seed=0, on_round=None):
    """
    Play seeded rounds between four policy factories.

    Args:
        factories: one rng -> policy callable per seat
        rounds: number of rounds to play
        seed: master seed; the same seed replays the same rounds
        on_round: optional callable(round_index, ranking)

    Returns:
        One Counter per seat mapping finish place (0 = Tycoon) to how
        often that seat took it.
    """
    rng = random.Random(seed)
    places = [Counter() for _ in factories]
    for index in range(rounds):
        hands = deal(rng)
        policies = [factory(random.Random(rng.random()))
                    for factory in factories]
        ranking = play_round(hands, first_leader(hands), policies)
        for place, player in enumerate(ranking):
            places[player][place] += 1
        if on_round is not None:
            on_round(index, ranking)
    return places


def scoreboard(places, labels):
    """Human-readable table of a `compare` result."""
    rounds = sum(places[0].values())
    lines = [f"{'':16}" + ''.join(f"{name:>9}" for name in PLACE_NAMES)]
    for seat, label in enumerate(labels):
        cells = ''.join(f"{places[seat][p]/rounds:>9.0%}" for p in range(4))
        lines.append(f"{label:16}" + cells)
    return '\n'.join(lines)


if __name__ == '__main__':
    from GameLogic.SearchRecommender import SearchPolicy
    from GameLogic.Simulator import random_policy, recommender_policy

    factories = [lambda rng: SearchPolicy(samples=16, rng=rng),
                 lambda rng: recommender_policy,
                 lambda rng: recommender_policy,
                 random_policy]
    results = compare(factories, rounds=50, seed=1,
                      on_round=lambda i, r: print('.', end='', flush=True))
    print()
    print(scoreboard(results, ['search', 'heuristic', 'heuristic',
                               'random']))
