"""A small learned evaluator for candidate moves.

The net answers the same question a rollout answers — "what place do
I finish if I play this?" — from a Features.encode vector, in one
forward pass instead of a simulated round. Trained by distillation:
the rollout search's per-candidate average places are the targets
(see SelfPlay), so the net is a fast approximation of the search,
learned purely from self-play.

LearnedPolicy plays argmin of the predicted place. It slots into the
Simulator as a policy, and into SearchPolicy as a rollout policy.
"""

import torch
from torch import nn

from GameLogic.Features import FEATURE_COUNT, encode
from GameLogic.Recommender import _cost
from GameLogic.Rules import PASS, legal_moves


class PlaceNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(FEATURE_COUNT, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def train(features, places, epochs=30, batch_size=256, lr=1e-3,
          device=None, log=None):
    """
    Fit a PlaceNet on (state+move, average place) pairs.

    Args:
        features: list of Features.encode vectors
        places: matching average finish places 0..3 (scaled inside)
        log: optional callable(epoch, loss)

    Returns:
        The trained model, in eval mode on the CPU.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(places, dtype=torch.float32, device=device) / 3.0

    model = PlaceNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        permutation = torch.randperm(len(x), device=device)
        total = 0.0
        for start in range(0, len(x), batch_size):
            batch = permutation[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)
        if log is not None:
            log(epoch, total / len(x))

    return model.cpu().eval()


def save(model, path):
    torch.save(model.state_dict(), path)


def load(path):
    model = PlaceNet()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model.eval()


class LearnedPolicy:
    """A Simulator policy: argmin predicted place over the candidates.

    `wants_observation` marks it for SearchPolicy: passed as the
    rollout_policy, it receives the acting seat's full Observation of
    each determinized rollout world (rollout_observation) instead of
    a bare (hand, trick, revolution) — steering the rollouts with the
    same in-distribution inputs it was trained on. A context-blind
    variant (zeroed unseen/counts) was tried first and made the
    distilled next generation clearly weaker.
    """

    wants_observation = True

    def __init__(self, model):
        self.model = model

    def __call__(self, obs):
        hand = list(obs.hand)
        moves = legal_moves(hand, obs.trick, obs.revolution)
        if not moves:
            return PASS
        finishers = [m for m in moves if len(m) == len(hand)]
        if finishers:
            return min(finishers, key=lambda m: _cost(m, obs.revolution))

        candidates = list(moves)
        if obs.trick:
            candidates.append(PASS)
        with torch.no_grad():
            x = torch.tensor([encode(obs, move) for move in candidates],
                             dtype=torch.float32)
            predicted = self.model(x)
        return candidates[int(predicted.argmin())]
