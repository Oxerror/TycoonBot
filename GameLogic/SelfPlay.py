"""Self-play training pipeline: search plays itself, a net distills it.

Four SearchPolicies play seeded rounds against each other; every
sampled decision is tapped through the recorder as (situation,
candidate, average finish place over the determinized rollouts).
PolicyNet.train fits the evaluator on those pairs, and the arena
measures the result against the heuristic baseline.

Run the whole pipeline:

    .venv\\Scripts\\python -m GameLogic.SelfPlay [rounds]

writes the dataset to GameLogic/selfplay_data.npz (gitignored), the
model to GameLogic/policy_net.pt, and prints the arena scoreboard.
`--train-only` retrains from every saved dataset without regenerating;
`--chunk INDEX ROUNDS` generates one seeded chunk file and exits —
run chunks in separate processes so a crash (a rare interpreter
access violation was seen on long single-process runs) costs one
chunk instead of the whole dataset, then `--train-only` merges them.
"""

import random
import sys

from GameLogic.Features import encode
from GameLogic.SearchRecommender import SearchPolicy
from GameLogic.Simulator import deal, first_leader, play_round

MODEL_PATH = 'GameLogic/policy_net.pt'
DATA_PATH = 'GameLogic/selfplay_data.npz'
CHUNK_PATTERN = 'GameLogic/selfplay_chunk_{}.npz'
CHUNK_SEED_BASE = 1000


def generate_dataset(rounds=200, samples=12, seed=0, log=None):
    """
    Search-vs-search self-play, recording every sampled decision.

    Returns:
        (features, places): parallel lists — one entry per candidate
        move per decision, the target being that candidate's average
        finish place over the decision's determinizations.
    """
    rng = random.Random(seed)
    features, places = [], []

    def recorder(obs, candidates, averages):
        for move, average in zip(candidates, averages):
            features.append(encode(obs, move))
            places.append(average)

    for index in range(rounds):
        hands = deal(rng)
        policies = [SearchPolicy(samples=samples,
                                 rng=random.Random(rng.random()),
                                 recorder=recorder)
                    for _ in range(4)]
        play_round(hands, first_leader(hands), policies)
        if log is not None and (index + 1) % 10 == 0:
            log(index + 1, len(features))

    return features, places


def generate_chunk(index, rounds):
    """One seeded slice of the dataset in its own file; separate
    processes per chunk keep a crash from costing everything."""
    import numpy as np

    features, places = generate_dataset(rounds=rounds,
                                        seed=CHUNK_SEED_BASE + index)
    path = CHUNK_PATTERN.format(index)
    np.savez_compressed(path,
                        features=np.array(features, dtype=np.float32),
                        places=np.array(places, dtype=np.float32))
    print(f"chunk {index}: {len(features)} samples -> {path}")


def _load_saved_datasets(np):
    import glob

    features, places = [], []
    for path in [DATA_PATH] + sorted(glob.glob(CHUNK_PATTERN.format('*'))):
        try:
            stored = np.load(path)
        except FileNotFoundError:
            continue
        features.extend(stored['features'].tolist())
        places.extend(stored['places'].tolist())
        print(f"  loaded {len(stored['places'])} samples from {path}")
    return features, places


def main(rounds=200, train_only=False):
    import numpy as np

    from GameLogic.Arena import compare, scoreboard
    from GameLogic.PolicyNet import LearnedPolicy, save, train
    from GameLogic.Simulator import recommender_policy

    if train_only:
        features, places = _load_saved_datasets(np)
        if not features:
            raise SystemExit("No saved datasets found")
        print(f"Loaded {len(features)} samples total")
    else:
        print(f"Self-play: {rounds} rounds of search vs search...")
        features, places = generate_dataset(
            rounds=rounds,
            log=lambda r, n: print(f"  {r}/{rounds} rounds, {n} samples"))
        np.savez_compressed(DATA_PATH,
                            features=np.array(features, dtype=np.float32),
                            places=np.array(places, dtype=np.float32))
        print(f"Dataset saved to {DATA_PATH}")

    print(f"Training on {len(features)} decision samples...")
    model = train(features, places,
                  log=lambda e, loss: print(f"  epoch {e}: {loss:.4f}")
                  if e % 5 == 0 else None)
    save(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    print("Arena: learned policy vs three heuristics (100 rounds)...")
    policy = LearnedPolicy(model)
    results = compare([lambda rng: policy,
                       lambda rng: recommender_policy,
                       lambda rng: recommender_policy,
                       lambda rng: recommender_policy], rounds=100, seed=7)
    print(scoreboard(results, ['learned', 'heuristic', 'heuristic',
                               'heuristic']))


if __name__ == '__main__':
    if '--chunk' in sys.argv:
        position = sys.argv.index('--chunk')
        generate_chunk(int(sys.argv[position + 1]),
                       int(sys.argv[position + 2]))
    elif '--train-only' in sys.argv:
        main(train_only=True)
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
