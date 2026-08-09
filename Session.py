"""Per-frame game-following logic.

Shared by the live capture loop (VideoStream) and the offline replay
tool (Replay), so both follow the game with exactly the same brain:
recognize the hand and current trick, track plays into GameState,
verify against the status bar, detect whose turn it is and suggest a
move on ours.
"""

from GameLogic.Card import Card
from GameLogic.GameState import DECK_SIZE, GameState, validate_start_hand
from GameLogic.HandReader import detections_to_cards, hand_is_ordered
from GameLogic.PlayTracker import PlayTracker
from GameLogic.Recommender import recommend
from CardsLeftReader import read_cards_left_detailed
from ImageRecognition import (HAND_MATCH_PARAMS, get_recognizer,
                              read_play_field, read_revolution_indicator)
from ScreenCapture import cropRegion
from StatusBarReader import read_status_bar


class TycoonSession:
    def __init__(self, config):
        self.config = config
        self.recognizer = get_recognizer()
        self.game_state = None
        self.tracker = None
        self.diverged_frames = 0
        self.previous_bar = None

    def _start_tracking(self, bar_counts, trick, cards, player_count):
        """Begin tracking fresh (first frame or a new round)."""
        self.game_state = GameState.from_status_bar(bar_counts)
        self.tracker = PlayTracker(self.game_state)
        self.tracker.update(trick, cards, player_count)
        self.diverged_frames = 0

    def _resync(self, bar_counts, trick, cards, player_count):
        """Re-adopt the bar mid-round, keeping hand/revolution knowledge."""
        self.game_state = GameState.from_status_bar(bar_counts)
        self.tracker.resync(self.game_state, trick, cards, player_count)
        self.diverged_frames = 0

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

        bar_counts = read_status_bar(frame)
        counters, _, active_player = read_cards_left_detailed(frame)
        all_counts = list(counters.values())
        round_start = None not in all_counts and sum(all_counts) == DECK_SIZE

        trick = read_play_field(play_field)
        if trick:
            messages.append(f"Current trick: {trick}")

        detections = []
        cards = None
        if self._needs_hand_reading(counters, active_player, round_start):
            detections = self.recognizer.template_match(current_hand, **HAND_MATCH_PARAMS)
            self.recognizer.refine_suit_detections(current_hand, detections)
            cards = detections_to_cards(detections)
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
            self.tracker.revolution = read_revolution_indicator(frame)
            if self.tracker.revolution:
                messages.append("REVOLUTION is active - strength order is flipped")
            if active_player == 'player':
                own_hand = (self.tracker.known_hand_cards()
                            if self.tracker.known_hand else cards)
                if own_hand:
                    move = recommend(own_hand, trick, self.tracker.revolution)
                    messages.append(f"YOUR TURN - suggested play: "
                                    f"{list(move) if move else 'PASS'}")
            elif active_player is not None:
                messages.append(f"Waiting: {active_player} opponent is playing")

        self.previous_bar = bar_counts
        return messages, detections, (current_hand, play_field)
