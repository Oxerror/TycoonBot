"""Turn-by-turn engine for one round of Tycoon.

Rules.py knows which sets beat which; this engine adds the flow of a
round: whose turn it is, what passing does, when a trick ends and who
leads the next one, and the order in which players finish.

Pass semantics (the one part not directly confirmed in-game yet):
passing locks a player out for the rest of the trick, and the trick
ends when every other player holding cards has passed — the last
player to lay down cards wins it and leads the next. This matches the
written rules of the game (xanthir.com/b5_y0: a player "passes for
the rest of the hand"; the pile clears when "three players have
passed") and the persistent per-player "Pass" speech bubbles on
screen, which mark a state rather than a one-turn event. If live
observation ever contradicts this, `step`'s PASS branch is the only
place encoding it.

The trick winner leads the next trick; when the winner finished on
that trick, the lead moves to the next player clockwise still holding
cards. Immediate winners (a set of 8s, the Wonder, the 3-Spade
Reversal) end the trick on the spot. Four-card sets toggle the
revolution for the rest of the round.

Usage: construct with the dealt hands and the leader, then repeatedly
ask `current` who acts and feed their move to `step` (PASS or a tuple
of Cards). `step` returns the events it caused, e.g. for a display or
a learning loop.
"""

from GameLogic.Rules import (PASS, causes_revolution, contains_wonder,
                             is_spade_three_counter, legal_moves,
                             move_strength, wins_trick_immediately)


class TrickEngine:
    def __init__(self, hands, leader, revolution=False, trick=(),
                 last_player=None, passed=()):
        """
        Args:
            hands: one iterable of Cards per player, in seating order
            leader: index of the player to act first
            revolution: True when a revolution is already active
            trick: a set already on the table, for resuming mid-trick
                (e.g. a search rollout from the live game's position)
            last_player: who laid down that set
            passed: players already locked out of the current trick
        """
        self.hands = [list(hand) for hand in hands]
        self.player_count = len(self.hands)
        self.trick = tuple(trick)
        if self.trick and last_player is None:
            raise ValueError("A resumed trick needs its owner (last_player)")
        self.last_player = last_player
        self.passed = set(passed)
        # A resumed round may start with players who already went out;
        # they occupy the top places (their order among themselves is
        # unknowable here and irrelevant to the players still in).
        self.finished = [p for p, hand in enumerate(self.hands) if not hand]
        self.revolution = revolution
        self.current = leader

    def round_over(self):
        return len(self.finished) == self.player_count

    def ranking(self):
        """Players in finish order: Tycoon first, Beggar last."""
        if not self.round_over():
            raise ValueError("The round is still running")
        return list(self.finished)

    def legal(self):
        """Legal sets for the current player (PASS is additionally
        allowed whenever there is a trick to answer)."""
        return legal_moves(self.hands[self.current], self.trick,
                           self.revolution)

    def step(self, move):
        """
        Let the current player act.

        Args:
            move: a tuple of Cards from the player's hand, or PASS

        Returns:
            List of event tuples in the order they happened:
            ('play', player, move), ('pass', player),
            ('revolution', active), ('finished', player),
            ('trick_won', player), ('round_over', ranking).

        Raises:
            ValueError: on illegal moves, a leading PASS, or when the
                round is already over.
        """
        if self.round_over():
            raise ValueError("The round is over")
        player = self.current
        events = []

        if move == PASS:
            if not self.trick:
                raise ValueError("The trick leader must play")
            self.passed.add(player)
            events.append(('pass', player))
        else:
            move = tuple(move)
            if not self._beats_trick(move):
                raise ValueError(f"Illegal move {move} on {self.trick}")
            self._take_from_hand(player, move)
            answered = self.trick
            self.trick = move
            self.last_player = player
            events.append(('play', player, move))

            if causes_revolution(move):
                self.revolution = not self.revolution
                events.append(('revolution', self.revolution))

            if not self.hands[player]:
                self.finished.append(player)
                events.append(('finished', player))
                if self._maybe_finish_round(events):
                    return events

            if wins_trick_immediately(move, answered):
                self._end_trick(events)
                return events

        if self._no_one_can_answer():
            self._end_trick(events)
        else:
            self._advance()
        return events

    def _beats_trick(self, move):
        if not self.trick:
            return True
        if contains_wonder(move):
            return True
        if is_spade_three_counter(move, self.trick):
            return True
        return (len(move) == len(self.trick)
                and (move_strength(move, self.revolution)
                     > move_strength(self.trick, self.revolution)))

    def _take_from_hand(self, player, move):
        hand = self.hands[player]
        for card in move:
            for i, held in enumerate(hand):
                if held is card:
                    del hand[i]
                    break
            else:
                # Fall back to rank+suit (Card.__eq__ is rank-only).
                for i, held in enumerate(hand):
                    if held.rank == card.rank and held.suit == card.suit:
                        del hand[i]
                        break
                else:
                    raise ValueError(f"{card} is not in player {player}'s hand")

    def _contenders(self):
        """Players still able to act in this trick."""
        return [p for p in range(self.player_count)
                if self.hands[p] and p not in self.passed]

    def _no_one_can_answer(self):
        return all(p == self.last_player for p in self._contenders())

    def _end_trick(self, events):
        winner = self.last_player
        events.append(('trick_won', winner))
        self.trick = ()
        self.passed = set()
        self.last_player = None
        if self.hands[winner]:
            self.current = winner
        else:
            self.current = self._next_with_cards(winner)

    def _advance(self):
        player = self.current
        for _ in range(self.player_count):
            player = (player + 1) % self.player_count
            if self.hands[player] and player not in self.passed:
                self.current = player
                return
        raise RuntimeError("No player can act")

    def _next_with_cards(self, player):
        for _ in range(self.player_count):
            player = (player + 1) % self.player_count
            if self.hands[player]:
                return player
        raise RuntimeError("No player holds cards")

    def _maybe_finish_round(self, events):
        remaining = [p for p in range(self.player_count) if self.hands[p]]
        if len(remaining) != 1:
            return False
        self.finished.append(remaining[0])
        events.append(('round_over', tuple(self.finished)))
        return True
