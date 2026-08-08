"""Heuristic play recommendation, version 1.

Deliberately simple — the goal is a sane baseline on top of the rules
engine that a future search- or learning-based recommender must beat:

- A move that empties the hand wins the round: always play it.
- Leading: dump the weakest set, preferring to play a whole set over
  breaking it up.
- Following: win with the cheapest sufficient set. Jokers carry a
  penalty so they are saved for when nothing else wins, and the Wonder
  even more so. Rather than spending those on an unimportant trick,
  pass — unless the hand is nearly empty, when winning tricks (and the
  lead) matters more than saving power cards.
"""

from GameLogic.Card import Rank
from GameLogic.Rules import (PASS, contains_wonder, legal_moves,
                             move_strength, set_rank)

# Above this cost a move spends power cards (Joker/Wonder) we would
# rather keep; see _cost.
POWER_THRESHOLD = 40

# With this few cards left, winning tricks beats saving power cards.
ENDGAME_HAND_SIZE = 3


def _cost(move, revolution):
    cost = move_strength(move, revolution)
    cost += 40 * sum(1 for card in move if card.rank == Rank.JOKER)
    if contains_wonder(move):
        cost += 100
    return cost


def recommend(hand, trick=(), revolution=False):
    """
    Suggest a move for the current situation.

    Args:
        hand: the own cards (Cards; suitless placeholders are fine)
        trick: the set currently on the table, empty when leading
        revolution: True while a revolution is active

    Returns:
        A move (tuple of Cards) or PASS.
    """
    hand = list(hand)
    moves = legal_moves(hand, trick, revolution)
    if not moves:
        return PASS

    finishers = [m for m in moves if len(m) == len(hand)]
    if finishers:
        return min(finishers, key=lambda m: _cost(m, revolution))

    if not trick:
        # Lead with the weakest set; prefer playing sets whole.
        return min(moves, key=lambda m: (_cost(m, revolution), -len(m)))

    cheapest = min(moves, key=lambda m: _cost(m, revolution))
    if _cost(cheapest, revolution) < POWER_THRESHOLD:
        return cheapest

    # Only power cards could win this trick.
    if len(hand) <= ENDGAME_HAND_SIZE:
        return cheapest
    return PASS
