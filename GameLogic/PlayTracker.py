"""Turn per-frame recognition results into observed plays.

Each frame the recognizer reports the cards of the current (bright)
trick on the table and the cards in the player's own hand. Comparing
consecutive readings tells us when new cards were played and by whom:

- A card is a new play the first time it shows up on the field this
  round (each rank/suit combination exists once per deck, jokers and
  wonders twice).
- If the new cards just disappeared from our own hand reading — or our
  own "Cards Left" counter dropped by exactly that many — we played
  them; they were never part of the unseen counts, so GameState is
  untouched.
- Otherwise an opponent played them and GameState.observe_opponent_play
  keeps the unseen bookkeeping current.

The tracker can also carry the complete own hand across frames: at
round start the status bar proves which cards the fan clipped away
(GameState.validate_start_hand), and set_known_hand stores the full 14
cards. Recovered cards without a readable suit are held as suitless
placeholders and upgraded once a reading shows their suit.

The very first update only seeds the baseline: cards already lying on
the table were played before tracking started and are covered by the
status-bar initialization of GameState.
"""

from collections import Counter

from GameLogic.Card import Card, Rank
from GameLogic.Rules import causes_revolution


class PlayTracker:
    def __init__(self, game_state):
        self.game_state = game_state
        self.seen_on_field = Counter()
        self.previous_hand = Counter()
        self.previous_player_count = None
        # (rank, suit) -> count; suit is None for cards recovered from
        # the bar whose suit was never readable. Empty until
        # set_known_hand is called.
        self.known_hand = Counter()
        # Flipped by every observed four-card set. Note: a tracker
        # re-created mid-round (bar re-sync) starts non-revolution.
        self.revolution = False
        self.started = False

    @staticmethod
    def _key(card):
        return (card.rank, card.suit)

    @staticmethod
    def _copies(rank):
        return 2 if rank in (Rank.JOKER, Rank.WONDER) else 1

    def set_known_hand(self, cards):
        """Store the complete own hand (round-start reading plus the
        cards recovered from the status bar)."""
        self.known_hand = Counter(self._key(card) for card in cards)

    def known_hand_cards(self):
        """The tracked own hand as Cards in the game's display order."""
        cards = [Card(rank, suit)
                 for (rank, suit), count in self.known_hand.items()
                 for _ in range(count)]
        return sorted(cards, key=lambda c: 0 if c.rank == Rank.WONDER
                      else c.rank.value)

    def _remove_from_known_hand(self, cards):
        for card in cards:
            key = self._key(card)
            if self.known_hand[key] > 0:
                self.known_hand[key] -= 1
            elif self.known_hand[(card.rank, None)] > 0:
                # A recovered card was played: its suit only became
                # known the moment it hit the table.
                self.known_hand[(card.rank, None)] -= 1
        self.known_hand = +self.known_hand

    def _upgrade_placeholders(self, current_hand):
        """A reading that shows more of a suited card than we know about
        resolves a suitless placeholder of the same rank."""
        for (rank, suit), count in current_hand.items():
            while (count > self.known_hand[(rank, suit)]
                   and self.known_hand[(rank, None)] > 0):
                self.known_hand[(rank, None)] -= 1
                self.known_hand[(rank, suit)] += 1
        self.known_hand = +self.known_hand

    def update(self, field_cards, hand_cards, player_cards_left=None):
        """
        Process one frame's readings.

        Args:
            field_cards: Cards of the current trick (read_play_field)
            hand_cards: Cards in the own hand (read_hand)
            player_cards_left: The own "Cards Left" counter, when
                readable. Used to attribute plays of cards the hand
                reading never showed (the clipped fan-edge cards).

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
            if self.known_hand:
                self._upgrade_placeholders(current_hand)
            self.previous_hand = current_hand
            self.previous_player_count = player_cards_left
            return []

        events = []
        if new_cards:
            left_hand = self.previous_hand - current_hand
            by_player = not Counter(self._key(c) for c in new_cards) - left_hand

            if (not by_player and player_cards_left is not None
                    and self.previous_player_count is not None
                    and self.previous_player_count - player_cards_left == len(new_cards)):
                # Our counter dropped by exactly these cards: we played
                # them even though the hand reading never showed them
                # (clipped fan-edge cards like the Wonder or Joker).
                by_player = True

            if by_player:
                self._remove_from_known_hand(new_cards)
            else:
                self.game_state.observe_opponent_play(new_cards)

            if causes_revolution(tuple(new_cards)):
                self.revolution = not self.revolution

            events.append({'cards': new_cards, 'by_player': by_player})

        if self.known_hand:
            self._upgrade_placeholders(current_hand)

        self.previous_hand = current_hand
        if player_cards_left is not None:
            self.previous_player_count = player_cards_left
        return events
