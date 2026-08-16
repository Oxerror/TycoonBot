"""Per-frame game-following logic.

Shared by the live capture loop (VideoStream) and the offline replay
tool (Replay), so both follow the game with exactly the same brain:
recognize the hand and current trick, track plays into GameState,
verify against the status bar, detect whose turn it is and suggest a
move on ours — including the button presses that would play it, which
the executor only logs unless it was explicitly built in act mode.

In act mode a turn is pressed at most once: the same turn is usually
seen on several consecutive frames while the game animates, and
re-pressing would double-play. The guard lifts once the turn moves on
(another player's arrow, our own play tracked, or tracking restarts);
until then held frames just report that the input is already out.
"""

import random

from CardsLeftReader import TURN_BUTTON_BAND
from GameLogic.Card import Card
from GameLogic.GameState import DECK_SIZE, GameState, validate_start_hand
from GameLogic.HandReader import hand_is_ordered
from GameLogic.PlayTracker import PlayTracker
from GameLogic.Recommender import recommend
from GameLogic.SearchRecommender import SearchPolicy
from GameLogic.Simulator import Observation
from FrameReader import FrameReader
from GameLogic.Rules import legal_moves
from InputExecutor import InputExecutor
from InputPlanner import cursor_stops, merge_into_fan, plan_move
from ScreenCapture import cropRegion


# Seating for the search rollouts, our seat first. Derived from the
# round-start turn sequences in the recorded session (clean first
# go-arounds read left->player->right and right->middle->left->player):
# play proceeds player -> right -> middle -> left.
TURN_ORDER = ('right', 'middle', 'left')


def _mask_turn_buttons(play_field, region):
    """Black out the Pass/Hint row inside the play-field crop.

    On our turn the game draws the button row across the bottom of the
    field; its white banners read as an event banner, which would blank
    the trick and make every suggestion assume we are leading. Masked
    rows are computed from the same band the turn detector matches.
    """
    fy1, fy2 = TURN_BUTTON_BAND[0], TURN_BUTTON_BAND[1]
    top, bottom = region['top'], region['bottom']
    height = play_field.shape[0]
    y1 = int((fy1 - top) / (bottom - top) * height)
    y2 = int((fy2 - top) / (bottom - top) * height)
    if y2 <= 0 or y1 >= height:
        return play_field
    masked = play_field.copy()
    masked[max(0, y1):y2, :] = 0
    return masked


class TycoonSession:
    def __init__(self, config, reader=None, executor=None, policy='search'):
        self.config = config
        self.reader = reader if reader is not None else FrameReader()
        self.executor = executor if executor is not None else InputExecutor()
        self.game_state = None
        self.tracker = None
        self.diverged_frames = 0
        self.previous_bar = None
        self.acted_this_turn = False
        # Seeded so a Replay of the same session suggests the same moves.
        self.search = SearchPolicy(samples=16, rng=random.Random(0))
        # 'net' answers each trusted turn with the trained evaluator in
        # one forward pass instead of the multi-second rollout search
        # (it beat search16 head-to-head in the arena); the untrusted-
        # state fallback below stays the heuristic either way.
        self.net_policy = None
        if policy == 'net':
            from GameLogic.SelfPlay import learned_rollout
            self.net_policy = learned_rollout()
        elif policy != 'search':
            raise ValueError(f"unknown policy {policy!r}")

    def _suggest(self, own_hand, trick, counters, passed_players):
        """Pick a move with the rollout search when the table state is
        trustworthy, falling back to the plain heuristic otherwise.

        The yellow bubble markers say who already passed this trick;
        the current set is attributed to the nearest predecessor who
        has not passed (whoever laid it down cannot carry the marker).
        """
        opponent_counts = [counters.get(name) for name in TURN_ORDER]
        unseen = self.game_state.unseen
        if (None not in opponent_counts
                and sum(opponent_counts) == self.game_state.total_unseen()):
            passed_seats = frozenset(TURN_ORDER.index(name) + 1
                                     for name in passed_players
                                     if name in TURN_ORDER)
            last_player = None
            if trick:
                last_player = len(TURN_ORDER)
                for seat in range(len(TURN_ORDER), 0, -1):
                    if seat not in passed_seats:
                        last_player = seat
                        break
            obs = Observation(seat=0,
                              hand=tuple(own_hand),
                              trick=tuple(trick),
                              revolution=self.tracker.revolution,
                              unseen=dict(unseen),
                              counts=(len(own_hand), *opponent_counts),
                              passed=passed_seats,
                              last_player=last_player)
            return (self.net_policy or self.search)(obs)
        return recommend(own_hand, trick, self.tracker.revolution,
                         unseen=unseen)

    def _start_tracking(self, bar_counts, trick, cards, player_count):
        """Begin tracking fresh (first frame or a new round)."""
        self.game_state = GameState.from_status_bar(bar_counts)
        self.tracker = PlayTracker(self.game_state)
        self.tracker.update(trick, cards, player_count)
        self.diverged_frames = 0
        self.acted_this_turn = False

    def _resync(self, bar_counts, trick, cards, player_count):
        """Re-adopt the bar mid-round, keeping hand/revolution knowledge."""
        self.game_state = GameState.from_status_bar(bar_counts)
        self.tracker.resync(self.game_state, trick, cards, player_count)
        self.diverged_frames = 0
        self.acted_this_turn = False

    def _needs_hand_reading(self, counters, active_player, round_start):
        """The hand read costs ~1.5s; opponents play faster than that,
        so it only runs when the own hand could have changed or is
        about to matter."""
        if self.tracker is None or not self.tracker.known_hand:
            return True
        if round_start or active_player == 'player':
            return True
        return counters['player'] != self.tracker.previous_player_count

    def process_frame(self, frame):
        """
        Follow one frame of the game.

        Returns:
            Tuple (messages, detections, regions): messages is the list
            of user-facing status lines; detections are the raw hand
            detections (for drawing, empty on frames that skip the hand
            read); regions is (hand, field) crops.
        """
        messages = []

        play_field = cropRegion(frame, self.config['play_field'])
        current_hand = cropRegion(frame, self.config['hand_region'])

        bar_counts = self.reader.bar(frame)
        counters, _, active_player, passed_players = self.reader.counters(frame)
        all_counts = list(counters.values())
        round_start = None not in all_counts and sum(all_counts) == DECK_SIZE

        opponent_passes = [p for p in passed_players if p != 'player']
        if opponent_passes:
            messages.append(f"Passed this trick: {', '.join(opponent_passes)}")

        field_view = play_field
        if active_player == 'player':
            field_view = _mask_turn_buttons(play_field,
                                            self.config['play_field'])
        banner, trick = self.reader.field(field_view)
        if banner:
            # Event banners (All Pass, 8 Stop, Done, ...) cover the
            # field and fool the card templates; skip this reading.
            messages.append("Event banner on the field - trick reading skipped")
        elif trick:
            messages.append(f"Current trick: {trick}")

        detections = []
        cards = None
        if self._needs_hand_reading(counters, active_player, round_start):
            detections, cards = self.reader.hand(current_hand)
            messages.append(f"Hand: {cards}")
            if not hand_is_ordered(cards):
                # The game always displays the hand sorted, so an unordered
                # reading proves at least one card was misrecognized.
                messages.append("WARNING: hand reading is out of display order - likely a misread")

        if bar_counts is None:
            messages.append("Status bar: not visible")
            return messages, detections, (current_hand, play_field)

        bar_total = sum(bar_counts.values())
        opponent_counts = [counters[k] for k in ('left', 'middle', 'right')]
        # A bar reading that matches the opponents' counters is
        # self-consistent ground truth; a mid-animation misread is not.
        bar_trusted = (None not in opponent_counts
                       and sum(opponent_counts) == bar_total)

        if self.game_state is None or round_start:
            # A fresh round deals new cards: restart tracking from the
            # bar. Nothing has been played yet, so this loses nothing
            # (and it is idempotent across repeated round-start frames).
            starting = self.game_state is None
            self._start_tracking(bar_counts, trick, cards or [], counters['player'])
            if starting:
                messages.append(f"Tracking started: {self.game_state.total_unseen()} unseen cards")
        elif bar_trusted and bar_total > self.game_state.total_unseen():
            # Unseen cards only ever shrink within a round: an increase
            # means a new round started without a clean start frame.
            messages.append("New round detected - restarting tracking")
            self._start_tracking(bar_counts, trick, cards or [], counters['player'])
        else:
            try:
                for event in self.tracker.update(trick, cards, counters['player']):
                    who = 'we' if event['by_player'] else 'opponent'
                    if event['by_player']:
                        # Our play registered: the turn we acted on is over.
                        self.acted_this_turn = False
                    messages.append(f"Play observed ({who}): {event['cards']}")
            except ValueError as error:
                messages.append(f"ALARM: impossible play observed - {error}")

            mismatches = self.game_state.verify_against(bar_counts)
            if not mismatches:
                self.diverged_frames = 0
                messages.append("State verified: tracking matches the game")
            else:
                self.diverged_frames += 1
                diff = ', '.join(f"{r.name} {t}->{a}"
                                 for r, (t, a) in mismatches.items())
                if bar_trusted:
                    # Plays slipped between frames; the bar is the same
                    # public information we track, so adopt it quietly.
                    messages.append(f"Missed play(s), re-synced from the bar ({diff})")
                    self._resync(bar_counts, trick, cards, counters['player'])
                elif self.diverged_frames >= 2 and (bar_counts == self.previous_bar
                                                    or self.diverged_frames >= 4):
                    messages.append(f"ALARM: bot state diverged from the game ({diff})")
                    messages.append("Re-syncing from the status bar.")
                    self._resync(bar_counts, trick, cards, counters['player'])
                else:
                    messages.append(f"State mismatch this frame ({diff}), waiting one frame")

        if round_start and cards is not None:
            # Nothing has been played, so deck - bar must equal the own
            # hand: validates the reading and recovers the cards clipped
            # at the fan edges.
            missing, extra = validate_start_hand(cards, bar_counts)
            if extra:
                messages.append("WARNING: hand reading shows cards the bar rules out: "
                                + ", ".join(f"{r.name} x{n}" for r, n in extra.items()))
            else:
                recovered = [Card(rank) for rank, n in missing.items()
                             for _ in range(n)]
                self.tracker.set_known_hand(cards + recovered)
                messages.append(
                    f"Round start: full hand known: {self.tracker.known_hand_cards()}"
                    + (f" ({len(recovered)} recovered from the bar)" if recovered else ""))

        if self.tracker is not None:
            # The persistent badge is authoritative: it survives quad
            # plays whose cards were never readable on the field.
            self.tracker.revolution = self.reader.revolution(frame)
            if self.tracker.revolution:
                messages.append("REVOLUTION is active - strength order is flipped")
            if active_player == 'player':
                own_hand = (self.tracker.known_hand_cards()
                            if self.tracker.known_hand else cards)
                if own_hand:
                    move = self._suggest(own_hand, trick, counters,
                                         passed_players)
                    messages.append(f"YOUR TURN - suggested play: "
                                    f"{list(move) if move else 'PASS'}")
                    # cards was read this frame (own turns always are);
                    # the tracked hand fills in the clipped fan edges.
                    # The cursor only stops on cards the game leaves
                    # bright: those participating in some legal move.
                    try:
                        fan = merge_into_fan(cards or [], own_hand)
                        stops = cursor_stops(fan, legal_moves(
                            own_hand, trick, self.tracker.revolution))
                        plan = plan_move(fan, move, stops=stops)
                    except ValueError as error:
                        messages.append(f"WARNING: no input plan - {error}")
                    else:
                        if self.acted_this_turn:
                            messages.append("Input already sent this turn - "
                                            "waiting for the game to register it")
                        else:
                            sent = self.executor.execute(plan)
                            self.acted_this_turn = sent
                            messages.append(('Input sent: ' if sent
                                             else 'Planned input: ') + ' '.join(plan))
            elif active_player is not None:
                self.acted_this_turn = False
                messages.append(f"Waiting: {active_player} opponent is playing")

        self.previous_bar = bar_counts
        return messages, detections, (current_hand, play_field)
