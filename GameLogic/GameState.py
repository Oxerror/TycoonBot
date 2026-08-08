"""Track the knowable game state during a round of Tycoon.

The bot's core state is the number of unseen cards per rank — the
cards still hidden in the opponents' hands. (On both reference
screenshots the top status bar sums exactly to the opponents' "Cards
Left" counters, so the bar shows precisely this: your own hand and all
played cards are excluded.)

Tracking starts from a status bar reading and every observed opponent
play decrements the matching rank. Because the bar is public
information the game keeps showing anyway, it doubles as ground truth:
`verify_against` compares the tracked state to a fresh bar reading and
reports every rank where the bot's bookkeeping went wrong.
"""

from GameLogic.Card import Rank


class GameState:
    def __init__(self, unseen):
        """
        Args:
            unseen: dict mapping every Rank to the number of cards of
                that rank not yet seen by the bot.
        """
        missing = set(Rank) - set(unseen)
        if missing:
            raise ValueError(f"unseen is missing ranks: {sorted(r.name for r in missing)}")
        self.unseen = dict(unseen)

    @classmethod
    def from_status_bar(cls, counts):
        """Start tracking from a StatusBarReader.read() result."""
        return cls(counts)

    def observe_opponent_play(self, cards):
        """
        Record cards an opponent just played.

        Own plays must not be recorded: the own hand was never part of
        the unseen counts.

        Raises:
            ValueError: if a rank is played more often than the state
                considered unseen — the tracking (or recognition) has
                already gone wrong at that point.
        """
        for card in cards:
            if self.unseen[card.rank] <= 0:
                raise ValueError(
                    f"Impossible play observed: no unseen {card.rank.name} left")
            self.unseen[card.rank] -= 1

    def verify_against(self, bar_counts):
        """
        Compare the tracked state to a fresh status bar reading.

        Args:
            bar_counts: dict from StatusBarReader.read(), or None when
                the bar was not readable.

        Returns:
            None when the bar was not readable (nothing to verify),
            otherwise a dict {Rank: (tracked, actual)} for every rank
            that disagrees — empty when the bot did not mess up.
        """
        if bar_counts is None:
            return None

        return {
            rank: (self.unseen[rank], bar_counts[rank])
            for rank in Rank
            if self.unseen[rank] != bar_counts[rank]
        }

    def total_unseen(self):
        """Total cards still hidden in opponents' hands."""
        return sum(self.unseen.values())

    def __repr__(self):
        parts = ' '.join(f"{r.name}:{c}" for r, c in self.unseen.items())
        return f"GameState({parts})"
