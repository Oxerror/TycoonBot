"""Translate a recommended move into gamepad button presses.

The Tycoon selection UI is cursor-driven: the hand fans out at the
bottom of the screen, the D-pad moves a cursor across the playable
cards (greyed-out ones are skipped), CROSS lifts the card under the
cursor into the staged play, OPTIONS submits the staged cards and
TRIANGLE passes. The planner turns "play these cards" into that button
sequence, locating each card by its position in the recognized fan and
counting steps across the playable slots only.

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
OPTIONS = 'OPTIONS'

# Verified against the real game in the supervised session of
# 2026-08-16. The selection UI works like this:
# - The cursor is hidden when a turn starts; the first d-pad press is
#   consumed revealing it on the leftmost PLAYABLE card, without
#   stepping ('reveal_press', prepended to every selection plan).
# - The cursor only stops on playable cards: the game greys out every
#   card that participates in no legal move, the cursor skips them,
#   and wrap-around wraps within the playable subset. Brightness is
#   fixed for the whole turn - lifting cards never re-greys the rest -
#   so plan_move steps across `stops` (see cursor_stops), not raw fan
#   slots.
# - Selecting lifts a card in place (fan order and neighbours are
#   untouched), so stop indices stay valid while a multi-card set is
#   gathered.
# - The fan shows the cards exactly in recognized left-to-right order
#   (merge_into_fan slots unread cards in by the game's display sort).
# Caveat: the game consumes one pad press to switch control modes
# after any human mouse/keyboard input. Autonomous play never
# triggers it, but a supervisor touching the mouse mid-session costs
# the next plan its reveal press.
UI_ASSUMPTIONS = {
    'reveal_press': DPAD_RIGHT,  # consumed unhiding the cursor, no step
    'cursor_start': 0,           # the cursor reveals on the leftmost stop
    'cursor_wraps': True,        # stepping past either end wraps (within the stops)
    'select_button': CROSS,      # lifts/toggles the card under the cursor
    'confirm_button': OPTIONS,   # submits the lifted cards as the play
    'pass_button': TRIANGLE,     # passes the turn instantly
    'pass_needs_confirm': False, # no confirm press follows the pass button
    'deselect_button': CIRCLE,   # lowers every lifted card (abort helper, never planned)
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


def _participates(card, move_card):
    """Whether a fan card can be the move card on screen: same rank,
    and same suit when both suits are known (either side may be a
    suitless bar-recovered placeholder)."""
    if card.rank != move_card.rank:
        return False
    return (card.suit is None or move_card.suit is None
            or card.suit == move_card.suit)


def cursor_stops(fan, moves):
    """Fan indices the cursor can stop on, given the legal moves.

    The game greys out every card that participates in no legal move
    and the cursor skips them, so only the participating slots are
    cursor stops. Brightness is fixed for the whole turn; one call per
    turn is enough.
    """
    return [index for index, card in enumerate(fan)
            if any(_participates(card, move_card)
                   for move in moves for move_card in move)]


def plan_move(fan, move, ui=None, stops=None):
    """The button sequence that plays `move` on the displayed `fan`.

    Args:
        fan: Cards as displayed left to right (merge_into_fan output).
        move: tuple of Cards to play, or Rules.PASS (empty) to pass.
        ui: dict overriding individual UI_ASSUMPTIONS entries.
        stops: fan indices the cursor stops on (cursor_stops output).
            None means every slot - only true when leading, where
            every card is playable.

    Returns:
        List of button names; a backend presses them one by one. A
        reveal press unhides the cursor first, then the cards are
        selected in fan order, each leg taking the shorter cursor
        direction across the stops.

    Raises:
        ValueError: a move card has no playable fan slot — recognition
            and recommendation disagree, and pressing anything would
            be a guess.
    """
    ui = {**UI_ASSUMPTIONS, **(ui or {})}
    if not move:
        plan = [ui['pass_button']]
        if ui['pass_needs_confirm']:
            plan.append(ui['confirm_button'])
        return plan

    if not fan:
        raise ValueError("cannot plan a play on an empty fan")

    stops = list(range(len(fan))) if stops is None else sorted(stops)
    if not stops:
        raise ValueError("a move is wanted but no fan slot is playable")

    available = list(stops)
    targets = []
    for card in move:
        slot = _matching_slot(fan, available, card)
        if slot is None:
            raise ValueError(f"{card} has no playable slot in the "
                             f"recognized fan {fan}")
        available.remove(slot)
        targets.append(slot)

    plan = [ui['reveal_press']] if ui['reveal_press'] else []
    position = min(ui['cursor_start'], len(stops) - 1)
    for target in sorted(targets):
        stop = stops.index(target)
        plan.extend(_cursor_steps(position, stop, len(stops),
                                  ui['cursor_wraps']))
        plan.append(ui['select_button'])
        position = stop
    plan.append(ui['confirm_button'])
    return plan
