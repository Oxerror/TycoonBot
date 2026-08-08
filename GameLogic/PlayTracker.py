"""Turn per-frame recognition results into observed plays.

Each frame the recognizer reports the cards of the current (bright)
trick on the table and the cards in the player's own hand. Comparing
consecutive readings tells us when new cards were played and by whom:

- A card is a new play the first time it shows up on the field this
  round (each rank/suit combination exists once per deck, jokers twice).
- If the new cards just disappeared from our own hand, we played them —
  they were never part of the unseen counts, so GameState is untouched.
- Otherwise an opponent played them and GameState.observe_opponent_play
  keeps the unseen bookkeeping current.

The very first update only seeds the baseline: cards already lying on
the table were played before tracking started and are covered by the
status-bar initialization of GameState.
"""

from collections import Counter

from GameLogic.Card import Rank


class PlayTracker:
    def __init__(self, game_state):
        self.game_state = game_state
        self.seen_on_field = Counter()
        self.previous_hand = Counter()
        self.started = False

    @staticmethod
    def _key(card):
        return (card.rank, card.suit)

    @staticmethod
    def _copies(rank):
        return 2 if rank == Rank.JOKER else 1

    def update(self, field_cards, hand_cards):
        """
        Process one frame's readings.

        Args:
            field_cards: Cards of the current trick (read_play_field)
            hand_cards: Cards in the own hand (read_hand)

        Returns:
            List of play events, each {'cards': [...], 'by_player': bool}.
            Empty when nothing new happened.

        Raises:
            ValueError: propagated from GameState when an observed
                opponent play is impossible for the tracked state —
                recognition or bookkeeping has gone wrong.
        """
        current_hand = Counter(self._key(card) for card in hand_cards)

        new_cards = []
        for card in field_cards:
            key = self._key(card)
            if self.seen_on_field[key] < self._copies(card.rank):
                self.seen_on_field[key] += 1
                new_cards.append(card)

        if not self.started:
            # Baseline frame: whatever lies on the table predates tracking.
            self.started = True
            self.previous_hand = current_hand
            return []

        events = []
        if new_cards:
            left_hand = self.previous_hand - current_hand
            by_player = not Counter(self._key(c) for c in new_cards) - left_hand

            if not by_player:
                self.game_state.observe_opponent_play(new_cards)

            events.append({'cards': new_cards, 'by_player': by_player})

        self.previous_hand = current_hand
        return events
