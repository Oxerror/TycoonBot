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
`--train-only` retrains without regenerating; `--chunk INDEX ROUNDS`
generates one seeded chunk file and exits — run chunks in separate
processes so a crash (a rare interpreter access violation was seen
on long single-process runs) costs one chunk instead of the whole
dataset. Training always pools every saved dataset, base and ladder.

`--chunk INDEX N --matches` generates a chunk of N whole MATCHES
instead of isolated rounds: three rounds each with the between-round
card exchange, recording role-shaped decisions with match context
(roles, round index) — the closest offline data to the real game.
Match chunks live in selfplay_match_chunk_* and join the pool; the
loader skips any saved dataset whose feature width predates the
current encoding.

`--ladder` climbs one rung of the self-play ladder: the current
policy_net.pt steers the rollouts inside the generating search (the
LearnedPolicy replaces the heuristic, fed each rollout seat's full
view of the determinized world), which sharpens the recorded targets.
Ladder data lives in separate files (selfplay_ladder_*) and joins the
training pool alongside the heuristic-steered data; the retrained
model overwrites policy_net.pt — the previous rung stays recoverable
from git. Ladder rollouts are ~12x slower than heuristic ones, so run
the chunks concurrently.

Lessons from the first rung, both measured head-to-head over 400
arena rounds: steering the rollouts context-blind (zeroed
unseen/counts) distills into a clearly weaker net — the full
observation matters; and training on the ladder data alone loses to
the pooled data — nets calibrate to the play style that produced
their targets, so keep both styles in the pool.
"""

import random
import sys

from GameLogic.Features import FEATURE_COUNT, encode
from GameLogic.SearchRecommender import SearchPolicy
from GameLogic.Simulator import deal, first_leader, play_game, play_round

MODEL_PATH = 'GameLogic/policy_net.pt'
DATA_PATH = 'GameLogic/selfplay_data.npz'
CHUNK_PATTERN = 'GameLogic/selfplay_chunk_{}.npz'
CHUNK_SEED_BASE = 1000
LADDER_DATA_PATH = 'GameLogic/selfplay_ladder_data.npz'
LADDER_CHUNK_PATTERN = 'GameLogic/selfplay_ladder_chunk_{}.npz'
LADDER_SEED_BASE = 2000
MATCH_CHUNK_PATTERN = 'GameLogic/selfplay_match_chunk_{}.npz'
MATCH_SEED_BASE = 3000
ROUNDS_PER_MATCH = 3


def learned_rollout(path=MODEL_PATH):
    """The trained net as a SearchPolicy rollout policy — the hook
    that turns one trained rung into the next rung's generator. The
    returned LearnedPolicy wants_observation, so the search feeds it
    each rollout seat's full view of the determinized world."""
    from GameLogic.PolicyNet import LearnedPolicy, load

    return LearnedPolicy(load(path))


def generate_dataset(rounds=200, samples=12, seed=0, log=None,
                     rollout_policy=None):
    """
    Search-vs-search self-play, recording every sampled decision.

    Args:
        rollout_policy: steers the seats inside the search rollouts;
            None for the heuristic, learned_rollout() for a ladder rung.

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
                                 rollout_policy=rollout_policy,
                                 recorder=recorder)
                    for _ in range(4)]
        play_round(hands, first_leader(hands), policies)
        if log is not None and (index + 1) % 10 == 0:
            log(index + 1, len(features))

    return features, places


def generate_match_dataset(matches=50, samples=12, seed=0, log=None,
                           rollout_policy=None):
    """
    Search-vs-search self-play over whole matches: three rounds with
    the between-round card exchange, so the recorded decisions cover
    role-shaped hands (a stacked Tycoon, a stripped Beggar) and carry
    the match context (roles, round index) the exchange creates.
    The exchange's return choice stays the weakest-cards default.

    Returns:
        (features, places): as generate_dataset, one entry per
        candidate move per sampled decision across all rounds.
    """
    rng = random.Random(seed)
    features, places = [], []

    def recorder(obs, candidates, averages):
        for move, average in zip(candidates, averages):
            features.append(encode(obs, move))
            places.append(average)

    for index in range(matches):
        policies = [SearchPolicy(samples=samples,
                                 rng=random.Random(rng.random()),
                                 rollout_policy=rollout_policy,
                                 recorder=recorder)
                    for _ in range(4)]
        play_game(policies, ROUNDS_PER_MATCH, rng)
        if log is not None and (index + 1) % 5 == 0:
            log(index + 1, len(features))

    return features, places


def generate_chunk(index, rounds, ladder=False, matches=False):
    """One seeded slice of the dataset in its own file; separate
    processes per chunk keep a crash from costing everything."""
    import numpy as np

    if matches:
        seed, path = MATCH_SEED_BASE + index, MATCH_CHUNK_PATTERN.format(index)
        features, places = generate_match_dataset(matches=rounds, seed=seed)
    else:
        if ladder:
            import torch
            torch.set_num_threads(1)    # chunks run concurrently
            seed, path = (LADDER_SEED_BASE + index,
                          LADDER_CHUNK_PATTERN.format(index))
            rollout_policy = learned_rollout()
        else:
            seed, path = CHUNK_SEED_BASE + index, CHUNK_PATTERN.format(index)
            rollout_policy = None
        features, places = generate_dataset(rounds=rounds, seed=seed,
                                            rollout_policy=rollout_policy)
    np.savez_compressed(path,
                        features=np.array(features, dtype=np.float32),
                        places=np.array(places, dtype=np.float32))
    print(f"chunk {index}: {len(features)} samples -> {path}")


def _load_saved_datasets(np):
    """Every saved dataset, base and ladder pooled — the replay
    buffer. Nets trained on a single generation's data were weaker in
    the arena than the pool: heuristic-steered targets keep the net
    calibrated against heuristic play, net-steered targets against
    its own — mixing keeps both."""
    import glob

    paths = ([DATA_PATH] + sorted(glob.glob(CHUNK_PATTERN.format('*')))
             + [LADDER_DATA_PATH]
             + sorted(glob.glob(LADDER_CHUNK_PATTERN.format('*')))
             + sorted(glob.glob(MATCH_CHUNK_PATTERN.format('*'))))
    features, places = [], []
    for path in paths:
        try:
            stored = np.load(path)
        except FileNotFoundError:
            continue
        if stored['features'].shape[1] != FEATURE_COUNT:
            print(f"  SKIPPED {path}: {stored['features'].shape[1]} features "
                  f"(current encoding has {FEATURE_COUNT}) - regenerate it")
            continue
        features.extend(stored['features'].tolist())
        places.extend(stored['places'].tolist())
        print(f"  loaded {len(stored['places'])} samples from {path}")
    return features, places


def main(rounds=200, train_only=False, ladder=False):
    import numpy as np

    from GameLogic.Arena import compare, scoreboard
    from GameLogic.PolicyNet import LearnedPolicy, save, train
    from GameLogic.Simulator import recommender_policy

    if not train_only:
        rollout = 'learned' if ladder else 'heuristic'
        print(f"Self-play: {rounds} rounds of search vs search "
              f"({rollout} rollouts)...")
        features, places = generate_dataset(
            rounds=rounds,
            # A single-run seed clear of the chunk seeds, so a full run
            # and chunk files never duplicate deals in the pool.
            seed=LADDER_SEED_BASE - 1 if ladder else 0,
            rollout_policy=learned_rollout() if ladder else None,
            log=lambda r, n: print(f"  {r}/{rounds} rounds, {n} samples"))
        data_path = LADDER_DATA_PATH if ladder else DATA_PATH
        np.savez_compressed(data_path,
                            features=np.array(features, dtype=np.float32),
                            places=np.array(places, dtype=np.float32))
        print(f"Dataset saved to {data_path}")

    features, places = _load_saved_datasets(np)
    if not features:
        raise SystemExit("No saved datasets found")
    print(f"Loaded {len(features)} samples total")

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
    ladder = '--ladder' in sys.argv
    matches = '--matches' in sys.argv
    if '--chunk' in sys.argv:
        position = sys.argv.index('--chunk')
        generate_chunk(int(sys.argv[position + 1]),
                       int(sys.argv[position + 2]), ladder=ladder,
                       matches=matches)
    elif '--train-only' in sys.argv:
        main(train_only=True, ladder=ladder)
    else:
        rounds = next((int(a) for a in sys.argv[1:] if not a.startswith('-')),
                      200)
        main(rounds, ladder=ladder)
