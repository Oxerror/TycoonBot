"""Tycoon (Persona 5 Royal) move rules.

Rules encoded here (confirmed in-game and via the community guides):

- Strength order 3 < 4 < ... < King < Ace < 2; the Joker beats every
  normal card. Only equal-rank sets are playable (no runs), and the
  answer to a trick must match its card count with strictly higher
  strength.
- Jokers substitute for any rank inside a set; a set of only jokers
  counts as the strongest set of its size.
- Revolution: any four-card set flips the strength order of the normal
  ranks (3 strongest, 2 weakest) until countered by another four-card
  set. Jokers stay on top.
- 8-stop: a set of 8s wins the trick immediately; the player leads the
  next trick.
- 3-Spade Reversal: a single 3 of Spades beats a single Joker (only
  singles on both sides) and wins the trick immediately.
- Wonder: can always be played regardless of the trick, wins the trick
  immediately, and the player leads the next one.

Round flow (not encoded yet, needed for the future game simulator):
- The holder of the 3 of Diamonds leads round one; afterwards the
  Beggar leads. A trick is won when the other three players pass after
  the last play (any play resets the passing).
- Card exchange between rounds: the Beggar gives their two best cards
  to the Tycoon, who returns two freely chosen; the Poor gives their
  best card to the Rich, who returns one freely chosen. The Wonder is
  ignored when determining "best".

A move is a tuple of Cards; PASS is the empty tuple.
"""

from GameLogic.Card import Rank, Suit

PASS = ()

# Strength values: normal ranks use their game order (1..13); the
# specials sit far above so revolution's flip never touches them.
JOKER_STRENGTH = 100
WONDER_STRENGTH = 200


def effective_strength(rank, revolution=False):
    """Strength of a rank under the current (possibly flipped) order."""
    if rank == Rank.JOKER:
        return JOKER_STRENGTH
    if rank == Rank.WONDER:
        return WONDER_STRENGTH
    return 14 - rank.value if revolution else rank.value


def set_rank(cards):
    """The rank a set counts as (jokers assume the accompanying rank)."""
    ranks = {card.rank for card in cards if card.rank != Rank.JOKER}
    if not ranks:
        return Rank.JOKER
    if len(ranks) > 1:
        raise ValueError(f"Not a valid set: {cards}")
    return next(iter(ranks))


def move_strength(cards, revolution=False):
    return effective_strength(set_rank(cards), revolution)


def is_spade_three_counter(move, trick):
    """Single 3 of Spades against a single Joker (3-Spade Reversal)."""
    return (len(move) == 1 and len(trick) == 1
            and trick[0].rank == Rank.JOKER
            and move[0].rank == Rank.THREE
            and move[0].suit == Suit.SPADES)


def contains_wonder(move):
    return any(card.rank == Rank.WONDER for card in move)


def causes_revolution(move):
    """Any four-card set flips (or restores) the strength order."""
    return len(move) >= 4 and not contains_wonder(move)


def wins_trick_immediately(move, trick=()):
    """Wonder, a set of 8s and the 3-Spade Reversal end the trick at
    once; the player leads the next one."""
    if not move:
        return False
    if contains_wonder(move):
        return True
    if set_rank(move) == Rank.EIGHT:
        return True
    return is_spade_three_counter(move, trick)


def is_unbeatable(move, unseen, revolution=False):
    """
    True when no unseen cards can top this move.

    Args:
        move: the set to judge (tuple of Cards)
        unseen: dict {Rank: count} of cards still hidden in the
            opponents' hands (GameState.unseen)
        revolution: True while a revolution is active

    The counts pool all opponents, so this is conservative in the
    right direction: if the pooled counts cannot form a beating set,
    no single opponent can either. The suit-blind counts force one
    more conservative call: any unseen 3 might be the 3 of Spades, so
    a lone Joker is never unbeatable while a 3 is out.
    """
    if wins_trick_immediately(move):
        return True
    if unseen[Rank.WONDER] > 0:
        return False

    size = len(move)
    strength = move_strength(move, revolution)
    jokers = unseen[Rank.JOKER]

    if size == 1 and set_rank(move) == Rank.JOKER and unseen[Rank.THREE] > 0:
        return False
    if jokers >= size and JOKER_STRENGTH > strength:
        return False
    for rank in Rank:
        if rank in (Rank.JOKER, Rank.WONDER):
            continue
        if effective_strength(rank, revolution) <= strength:
            continue
        # A beating set needs at least one real card of the rank;
        # jokers fill the remaining slots.
        if unseen[rank] >= 1 and unseen[rank] + jokers >= size:
            return False
    return True


def _candidate_sets(hand):
    """Every distinct set the hand can form, one representative per
    strength-equivalent combination (single 3s keep their suit apart —
    the 3 of Spades plays a special role)."""
    jokers = [c for c in hand if c.rank == Rank.JOKER]
    wonders = [c for c in hand if c.rank == Rank.WONDER]

    by_rank = {}
    for card in hand:
        if card.rank in (Rank.JOKER, Rank.WONDER):
            continue
        by_rank.setdefault(card.rank, []).append(card)

    moves = []
    for rank, cards in by_rank.items():
        for size in range(1, len(cards) + len(jokers) + 1):
            for joker_count in range(0, min(len(jokers), size - 1) + 1):
                real_count = size - joker_count
                if real_count < 1 or real_count > len(cards):
                    continue
                if rank == Rank.THREE and size == 1:
                    # Emit each suit: only the 3 of Spades counters a Joker.
                    for card in cards:
                        moves.append((card,))
                    break
                moves.append(tuple(cards[:real_count]) + tuple(jokers[:joker_count]))

    # Sets of only jokers are the strongest of their size.
    for size in range(1, len(jokers) + 1):
        moves.append(tuple(jokers[:size]))

    if wonders:
        moves.append((wonders[0],))

    return moves


def legal_moves(hand, trick=(), revolution=False):
    """
    All moves the hand may answer the current trick with.

    Args:
        hand: iterable of Cards (suitless placeholders are fine; only
            the 3 of Spades rule cares about a suit)
        trick: Cards of the set currently on the table; empty when
            leading, in which case every candidate set is legal
        revolution: True while a revolution is active

    Returns:
        List of moves (tuples of Cards). PASS is always available when
        not leading and is not included in the list.
    """
    candidates = _candidate_sets(list(hand))
    if not trick:
        return candidates

    trick_value = move_strength(trick, revolution)
    moves = []
    for move in candidates:
        if contains_wonder(move):
            moves.append(move)
        elif is_spade_three_counter(move, trick):
            moves.append(move)
        elif (len(move) == len(trick)
              and move_strength(move, revolution) > trick_value):
            moves.append(move)
    return moves
