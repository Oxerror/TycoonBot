"""Determinized rollout search: recommendation by simulation.

The heuristic recommender judges a move by looking at it; this one
judges it by playing it out. The hidden cards are known only as
counts, so the search samples determinizations — concrete deals of
the unseen cards to the opponents, respecting their Cards-Left
counters — and plays every candidate move to the end of the round in
each sampled world, with the fast heuristic steering all four seats.
The move with the best average finishing place wins.

Sampling is rank-exact but suit-blind (the unseen counts carry no
suits), so a determinization may occasionally hand an opponent a
3 of Spades that was in truth already played. Suits only matter for
the 3-Spade Reversal; the error is noise, not bias.

The same determinizations are reused across all candidate moves of a
decision (common random numbers), so candidates are compared on the
same worlds and the sample count can stay small.
"""

import random

from GameLogic.Card import Card, Rank, Suit
from GameLogic.Recommender import _cost, guaranteed_win, recommend
from GameLogic.Rules import PASS, legal_moves
from GameLogic.Simulator import Observation
from GameLogic.TrickEngine import TrickEngine

SUITS = tuple(Suit)


def rollout_observation(engine, seat, roles=None, round_index=0):
    """The acting seat's full view of a determinized rollout world.

    Inside a rollout every hidden card sits in some engine hand, so
    the seat's unseen counts are simply the other hands pooled — an
    Observation exactly like the live one, letting observation
    policies (the learned net, the recommender) steer rollouts with
    the same in-distribution inputs they were built for. The match
    context (roles, round) is the root decision's — a rollout finishes
    the same round.
    """
    unseen = {rank: 0 for rank in Rank}
    for other, hand in enumerate(engine.hands):
        if other != seat:
            for card in hand:
                unseen[card.rank] += 1
    return Observation(seat=seat,
                       hand=tuple(engine.hands[seat]),
                       trick=engine.trick,
                       revolution=engine.revolution,
                       unseen=unseen,
                       counts=tuple(len(hand) for hand in engine.hands),
                       passed=frozenset(engine.passed),
                       last_player=engine.last_player,
                       roles=roles,
                       round_index=round_index)


class SearchPolicy:
    """A Simulator policy (Observation -> move) that searches.

    Args:
        samples: determinizations per decision — more is stronger and
            slower; each costs (candidates + 1) rollouts... roughly
            samples * candidates round playouts per decision.
        max_candidates: cap on evaluated moves, cheapest-first; the
            heuristic cost order keeps the plausible moves in front.
        rng: seeded random.Random for reproducible play.
        rollout_policy: steers every seat inside the rollouts; default
            is the fast heuristic recommender. Either a plain
            (hand, trick, revolution) -> move callable, or — when it
            carries a truthy `wants_observation` attribute — an
            Observation -> move policy fed the acting seat's full view
            of the determinized world via rollout_observation (this is
            how the learned net steers rollouts without going blind).
        recorder: optional callable(obs, candidates, average_places)
            invoked after each sampled decision — the self-play
            training data tap.
    """

    def __init__(self, samples=16, max_candidates=12, rng=None,
                 rollout_policy=None, recorder=None):
        self.samples = samples
        self.max_candidates = max_candidates
        self.rng = rng if rng is not None else random.Random()
        self.rollout_policy = (rollout_policy if rollout_policy is not None
                               else recommend)
        self._rollout_wants_obs = getattr(self.rollout_policy,
                                          'wants_observation', False)
        self.recorder = recorder

    def __call__(self, obs):
        hand = list(obs.hand)
        moves = legal_moves(hand, obs.trick, obs.revolution)
        if not moves:
            return PASS

        # Certain outcomes need no sampling.
        finishers = [m for m in moves if len(m) == len(hand)]
        if finishers:
            return min(finishers, key=lambda m: _cost(m, obs.revolution))
        sure_win = guaranteed_win(hand, moves, obs.unseen, obs.revolution)
        if sure_win is not None:
            return sure_win

        candidates = sorted(moves, key=lambda m: _cost(m, obs.revolution))
        candidates = candidates[:self.max_candidates]
        if obs.trick:
            candidates.append(PASS)
        if len(candidates) == 1:
            return candidates[0]

        totals = [0.0] * len(candidates)
        for _ in range(self.samples):
            worlds = self._deal_unseen(obs)
            for i, move in enumerate(candidates):
                totals[i] += self._rollout(obs, move, worlds)

        if self.recorder is not None and self.samples:
            self.recorder(obs, candidates,
                          [total / self.samples for total in totals])

        best = min(range(len(candidates)), key=lambda i: totals[i])
        return candidates[best]

    def _deal_unseen(self, obs):
        """One concrete world: the unseen cards split among the other
        seats according to their hand sizes."""
        cards = []
        for rank, count in obs.unseen.items():
            for copy in range(count):
                suit = (None if rank in (Rank.JOKER, Rank.WONDER)
                        else SUITS[copy % len(SUITS)])
                cards.append(Card(rank, suit))
        self.rng.shuffle(cards)

        hands = {}
        start = 0
        for seat, count in enumerate(obs.counts):
            if seat == obs.seat:
                continue
            hands[seat] = cards[start:start + count]
            start += count
        return hands

    def _rollout(self, obs, move, worlds):
        """Play the candidate move in the sampled world and let the
        heuristic finish the round for everyone. Returns the own
        finishing place, 0 (Tycoon) to 3 (Beggar)."""
        hands = [None] * len(obs.counts)
        hands[obs.seat] = list(obs.hand)
        for seat, cards in worlds.items():
            hands[seat] = cards

        engine = TrickEngine(hands, leader=obs.seat,
                             revolution=obs.revolution, trick=obs.trick,
                             last_player=obs.last_player, passed=obs.passed)
        engine.step(move if move else PASS)
        while not engine.round_over():
            player = engine.current
            if self._rollout_wants_obs:
                engine.step(self.rollout_policy(
                    rollout_observation(engine, player, roles=obs.roles,
                                        round_index=obs.round_index)))
            else:
                engine.step(self.rollout_policy(engine.hands[player],
                                                engine.trick,
                                                engine.revolution))
        return engine.ranking().index(obs.seat)
