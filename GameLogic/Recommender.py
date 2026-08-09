"""Heuristic play recommendation, version 2: unseen-aware.

The baseline heuristics from v1 remain — the goal is still a sane
baseline a future search- or learning-based recommender must beat:

- A move that empties the hand wins the round: always play it.
- Leading: dump the weakest set, preferring to play a whole set over
  breaking it up.
- Following: win with the cheapest sufficient set. Jokers carry a
  penalty so they are saved for when nothing else wins, and the Wonder
  even more so. Rather than spending those on an unimportant trick,
  pass — unless the hand is nearly empty, when winning tricks (and the
  lead) matters more than saving power cards.

Version 2 adds knowledge of the unseen cards (GameState.unseen): a
move nobody can top is worth far more than its rank suggests. The big
win is the guaranteed-finish search — when the hand splits into sets
that can be led out back to back with every set but the last
unbeatable, the round is mathematically won, so play that line now
even if it spends a Joker the old heuristics would have hoarded.
"""

from GameLogic.Card import Rank
from GameLogic.Rules import (PASS, causes_revolution, contains_wonder,
                             is_unbeatable, legal_moves, move_strength,
                             set_rank)

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


def _without(cards, move):
    """The hand minus the played cards, matched by identity: Card.__eq__
    compares rank only and would strip the wrong suit (the 3 of Spades
    matters)."""
    rest = list(cards)
    for played in move:
        for i, card in enumerate(rest):
            if card is played:
                del rest[i]
                break
        else:
            rest.remove(played)
    return rest


def _signature(cards, revolution):
    counts = {}
    for card in cards:
        counts[card.rank] = counts.get(card.rank, 0) + 1
    return (tuple(sorted((r.value, n) for r, n in counts.items())), revolution)


def _leads_out(cards, unseen, revolution, memo):
    """True when the cards can be led out back to back with no chance
    of interruption: every set unbeatable, except the very last which
    ends the round no matter what tops it."""
    if not cards:
        return True
    key = _signature(cards, revolution)
    if key in memo:
        return memo[key]

    result = False
    for move in legal_moves(cards, (), revolution):
        rest = _without(cards, move)
        if not rest:
            result = True
            break
        if not is_unbeatable(move, unseen, revolution):
            continue
        # Our own four-card set flips the order for the rest of the line.
        flipped = revolution != causes_revolution(move)
        if _leads_out(rest, unseen, flipped, memo):
            result = True
            break

    memo[key] = result
    return result


def guaranteed_win(hand, moves, unseen, revolution=False):
    """A legal move that starts an unstoppable lead-out of the whole
    hand, or None. Cheapest first so the pick looks natural when
    several lines win."""
    memo = {}
    for move in sorted(moves, key=lambda m: _cost(m, revolution)):
        if not is_unbeatable(move, unseen, revolution):
            continue
        rest = _without(hand, move)
        flipped = revolution != causes_revolution(move)
        if _leads_out(rest, unseen, flipped, memo):
            return move
    return None


def recommend(hand, trick=(), revolution=False, unseen=None):
    """
    Suggest a move for the current situation.

    Args:
        hand: the own cards (Cards; suitless placeholders are fine)
        trick: the set currently on the table, empty when leading
        revolution: True while a revolution is active
        unseen: dict {Rank: count} of cards still hidden in the
            opponents' hands (GameState.unseen), or None to fall back
            to the count-blind v1 heuristics

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

    if unseen is not None:
        winner = guaranteed_win(hand, moves, unseen, revolution)
        if winner is not None:
            return winner

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
