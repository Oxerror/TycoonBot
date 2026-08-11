"""Translate a recommended move into gamepad button presses.

The Tycoon selection UI is cursor-driven: the hand fans out at the
bottom of the screen, the D-pad moves a cursor across the cards, one
button lifts the card under the cursor into the staged play, another
submits the staged cards (passing has its own button). The planner
turns "play these cards" into that button sequence, locating each card
by its position in the recognized fan.

The buttons use PlayStation names (the game is a PlayStation title);
VirtualGamepad translates them to whatever pad it emulates.

Nothing here presses anything — the output is a plain list of button
names for InputExecutor to log or, once act mode is ever enabled, to
send through a backend.
"""

from bisect import bisect_right

from GameLogic.Card import Rank

DPAD_LEFT = 'DPAD_LEFT'
DPAD_RIGHT = 'DPAD_RIGHT'
CROSS = 'CROSS'
CIRCLE = 'CIRCLE'
SQUARE = 'SQUARE'
TRIANGLE = 'TRIANGLE'

# Every unverified belief about the selection UI lives in this one
# block, so the first supervised live session can correct the lot in
# one place. Beyond these entries the planner assumes: the fan shows
# the cards exactly in recognized left-to-right order (merge_into_fan
# slots unread cards in by the game's display sort), and selecting a
# card lifts it in place, so cursor indices stay valid while a
# multi-card set is gathered.
UI_ASSUMPTIONS = {
    'cursor_start': 0,           # the cursor rests on the leftmost card at turn start
    'cursor_wraps': True,        # stepping past either end wraps to the other
    'select_button': CROSS,      # lifts/toggles the card under the cursor
    'confirm_button': TRIANGLE,  # submits the lifted cards as the play
    'pass_button': SQUARE,       # passes the turn
    'pass_needs_confirm': False, # whether a confirm press must follow the pass button
}


def _display_key(card):
    """The game's hand sort: Wonder leftmost, then 3..2, Joker last."""
    return 0 if card.rank == Rank.WONDER else card.rank.value


def _matching_slot(fan, available, card):
    """Leftmost unused fan slot holding the card, or None.

    Exact (rank, suit) matches win; a rank-only match is accepted
    second, because either side may carry a suitless placeholder (a
    card recovered from the status bar whose suit was never readable)
    and a misread suit must not block selecting a card the tracker
    knows is there.
    """
    exact = [i for i in available
             if fan[i].rank == card.rank and fan[i].suit == card.suit]
    if exact:
        return exact[0]
    loose = [i for i in available if fan[i].rank == card.rank]
    if loose:
        return loose[0]
    return None


def merge_into_fan(recognized, hand):
    """Best guess of the full on-screen fan, left to right.

    In dense fans the outermost cards are clipped beyond recognition;
    the tracker still knows them, and they still occupy fan slots the
    cursor must step across. Cards of `hand` that the reading does not
    account for are slotted in where the game's display sort must put
    them (ties go after their visible rank-mates — unproven, but it
    only matters when selecting a specific suit out of equals).
    Recognized cards keep their observed positions even when `hand`
    disagrees: a misread label still marks a real slot.
    """
    fan = list(recognized)
    available = list(range(len(fan)))
    unplaced = []
    for card in hand:
        slot = _matching_slot(fan, available, card)
        if slot is None:
            unplaced.append(card)
        else:
            available.remove(slot)
    for card in sorted(unplaced, key=_display_key):
        keys = [_display_key(c) for c in fan]
        fan.insert(bisect_right(keys, _display_key(card)), card)
    return fan


def _cursor_steps(position, target, size, wraps):
    """D-pad presses moving the cursor from position to target."""
    if wraps:
        right = (target - position) % size
        left = (position - target) % size
        if right <= left:
            return [DPAD_RIGHT] * right
        return [DPAD_LEFT] * left
    delta = target - position
    if delta >= 0:
        return [DPAD_RIGHT] * delta
    return [DPAD_LEFT] * -delta


def plan_move(fan, move, ui=None):
    """The button sequence that plays `move` on the displayed `fan`.

    Args:
        fan: Cards as displayed left to right (merge_into_fan output).
        move: tuple of Cards to play, or Rules.PASS (empty) to pass.
        ui: dict overriding individual UI_ASSUMPTIONS entries.

    Returns:
        List of button names; a backend presses them one by one. The
        cards are selected in fan order, each leg taking the shorter
        cursor direction.

    Raises:
        ValueError: a move card has no fan slot — recognition and
            recommendation disagree, and pressing anything would be a
            guess.
    """
    ui = {**UI_ASSUMPTIONS, **(ui or {})}
    if not move:
        plan = [ui['pass_button']]
        if ui['pass_needs_confirm']:
            plan.append(ui['confirm_button'])
        return plan

    if not fan:
        raise ValueError("cannot plan a play on an empty fan")

    available = list(range(len(fan)))
    targets = []
    for card in move:
        slot = _matching_slot(fan, available, card)
        if slot is None:
            raise ValueError(f"{card} has no slot in the recognized fan {fan}")
        available.remove(slot)
        targets.append(slot)

    plan = []
    position = min(ui['cursor_start'], len(fan) - 1)
    for target in sorted(targets):
        plan.extend(_cursor_steps(position, target, len(fan),
                                  ui['cursor_wraps']))
        plan.append(ui['select_button'])
        position = target
    plan.append(ui['confirm_button'])
    return plan
