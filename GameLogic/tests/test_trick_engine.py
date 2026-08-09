import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from GameLogic.Card import Card, Rank, Suit
from GameLogic.Rules import PASS
from GameLogic.TrickEngine import TrickEngine


def c(rank, suit=None):
    return Card(rank, suit)


def hand(*cards):
    return list(cards)


def events_of(kind, events):
    return [e for e in events if e[0] == kind]


def make_engine(*hands, leader=0):
    """Pad every hand with filler so nobody finishes by accident."""
    return TrickEngine(hands, leader)


FILLER = [c(Rank.FOUR, Suit.CLUBS), c(Rank.FOUR, Suit.HEARTS)]


class TestTurnFlow:
    def test_leader_must_play(self):
        engine = make_engine(hand(c(Rank.FIVE, Suit.CLUBS), *FILLER),
                             hand(c(Rank.SIX, Suit.CLUBS), *FILLER),
                             hand(c(Rank.SEVEN, Suit.CLUBS), *FILLER),
                             hand(c(Rank.NINE, Suit.CLUBS), *FILLER))
        with pytest.raises(ValueError):
            engine.step(PASS)

    def test_turns_go_clockwise(self):
        engine = make_engine(hand(c(Rank.FIVE, Suit.CLUBS), *FILLER),
                             hand(c(Rank.SIX, Suit.CLUBS), *FILLER),
                             hand(c(Rank.SEVEN, Suit.CLUBS), *FILLER),
                             hand(c(Rank.NINE, Suit.CLUBS), *FILLER))
        engine.step((engine.hands[0][0],))
        assert engine.current == 1
        engine.step((engine.hands[1][0],))
        assert engine.current == 2

    def test_illegal_move_rejected(self):
        five = c(Rank.FIVE, Suit.CLUBS)
        four = c(Rank.FOUR, Suit.SPADES)
        engine = make_engine(hand(five, *FILLER),
                             hand(four, *FILLER),
                             hand(c(Rank.SEVEN, Suit.CLUBS), *FILLER),
                             hand(c(Rank.NINE, Suit.CLUBS), *FILLER))
        engine.step((five,))
        with pytest.raises(ValueError):
            engine.step((four,))

    def test_playing_a_card_not_in_hand_rejected(self):
        engine = make_engine(hand(c(Rank.FIVE, Suit.CLUBS), *FILLER),
                             hand(c(Rank.SIX, Suit.CLUBS), *FILLER),
                             hand(c(Rank.SEVEN, Suit.CLUBS), *FILLER),
                             hand(c(Rank.NINE, Suit.CLUBS), *FILLER))
        with pytest.raises(ValueError):
            engine.step((c(Rank.ACE, Suit.SPADES),))


class TestPassing:
    def make(self):
        return make_engine(
            hand(c(Rank.FIVE, Suit.CLUBS), c(Rank.KING, Suit.CLUBS), *FILLER),
            hand(c(Rank.SIX, Suit.CLUBS), *FILLER),
            hand(c(Rank.SEVEN, Suit.CLUBS), c(Rank.QUEEN, Suit.CLUBS), *FILLER),
            hand(c(Rank.NINE, Suit.CLUBS), *FILLER))

    def test_trick_ends_when_all_others_passed(self):
        engine = self.make()
        engine.step((engine.hands[0][0],))    # 0 plays the 5
        engine.step(PASS)                     # 1
        engine.step(PASS)                     # 2
        events = engine.step(PASS)            # 3 - only player 0 remains
        assert ('trick_won', 0) in events
        assert engine.trick == ()
        assert engine.current == 0

    def test_a_pass_locks_the_player_out_of_the_trick(self):
        engine = self.make()
        engine.step((engine.hands[0][0],))    # 0 plays the 5
        engine.step(PASS)                     # 1 passes - out for the trick
        engine.step((engine.hands[2][1],))    # 2 answers with the Queen
        engine.step(PASS)                     # 3 passes
        assert engine.current == 0            # 1 is skipped from now on
        events = engine.step((engine.hands[0][0],))  # 0 tops with the King
        assert engine.current == 2            # straight back to 2, not 1
        events = engine.step(PASS)            # 2 - everyone else is out
        assert ('trick_won', 0) in events

    def test_winner_leads_the_next_trick(self):
        engine = self.make()
        engine.step((engine.hands[0][0],))    # 0: 5
        engine.step((engine.hands[1][0],))    # 1: 6
        engine.step(PASS)                     # 2
        engine.step(PASS)                     # 3
        events = engine.step(PASS)            # 0
        assert ('trick_won', 1) in events
        assert engine.current == 1
        assert engine.passed == set()         # pass lock ends with the trick


class TestImmediateWins:
    def test_eight_stop_ends_the_trick(self):
        eight = c(Rank.EIGHT, Suit.CLUBS)
        engine = make_engine(hand(c(Rank.FIVE, Suit.CLUBS), *FILLER),
                             hand(eight, *FILLER),
                             hand(c(Rank.SEVEN, Suit.CLUBS), *FILLER),
                             hand(c(Rank.NINE, Suit.CLUBS), *FILLER))
        engine.step((engine.hands[0][0],))
        events = engine.step((eight,))
        assert ('trick_won', 1) in events
        assert engine.current == 1

    def test_wonder_ends_the_trick(self):
        wonder = c(Rank.WONDER)
        engine = make_engine(hand(c(Rank.TWO, Suit.CLUBS), *FILLER),
                             hand(wonder, *FILLER),
                             hand(c(Rank.SEVEN, Suit.CLUBS), *FILLER),
                             hand(c(Rank.NINE, Suit.CLUBS), *FILLER))
        engine.step((engine.hands[0][0],))
        events = engine.step((wonder,))
        assert ('trick_won', 1) in events

    def test_spade_three_counters_the_joker(self):
        joker = c(Rank.JOKER)
        spade_three = c(Rank.THREE, Suit.SPADES)
        engine = make_engine(hand(joker, *FILLER),
                             hand(spade_three, *FILLER),
                             hand(c(Rank.SEVEN, Suit.CLUBS), *FILLER),
                             hand(c(Rank.NINE, Suit.CLUBS), *FILLER))
        engine.step((joker,))
        events = engine.step((spade_three,))
        assert ('trick_won', 1) in events


class TestRevolution:
    def test_quad_toggles_and_flips_legality(self):
        nines = [c(Rank.NINE, s) for s in Suit]
        three = c(Rank.THREE, Suit.CLUBS)
        jack = c(Rank.JACK, Suit.CLUBS)
        engine = make_engine(hand(*nines, *FILLER),
                             hand(three, jack, *FILLER),
                             hand(c(Rank.SEVEN, Suit.CLUBS), *FILLER),
                             hand(c(Rank.TEN, Suit.CLUBS), *FILLER))
        events = engine.step(tuple(nines))
        assert ('revolution', True) in events
        assert engine.revolution
        # Trick continues: a quad is not an immediate winner.
        engine.step(PASS)                     # 1
        engine.step(PASS)                     # 2
        events = engine.step(PASS)            # 3
        assert ('trick_won', 0) in events
        engine.step((engine.hands[0][0],))    # 0 leads a filler 4
        with pytest.raises(ValueError):
            engine.step((jack,))              # higher rank now loses
        events = engine.step((three,))        # only the 3 tops a 4 now
        assert ('play', 1, (three,)) in events


class TestFinishing:
    def test_finish_order_and_round_end(self):
        engine = TrickEngine(
            [hand(c(Rank.FIVE, Suit.CLUBS)),
             hand(c(Rank.SIX, Suit.CLUBS)),
             hand(c(Rank.SEVEN, Suit.CLUBS)),
             hand(c(Rank.FOUR, Suit.CLUBS), c(Rank.NINE, Suit.CLUBS))],
            leader=0)
        events = engine.step((engine.hands[0][0],))   # 0 out
        assert ('finished', 0) in events
        events = engine.step((engine.hands[1][0],))   # 1 out
        assert ('finished', 1) in events
        events = engine.step((engine.hands[2][0],))   # 2 out
        assert ('finished', 2) in events
        assert ('round_over', (0, 1, 2, 3)) in events
        assert engine.ranking() == [0, 1, 2, 3]

    def test_lead_moves_on_when_the_winner_finished(self):
        engine = TrickEngine(
            [hand(c(Rank.TWO, Suit.CLUBS)),
             hand(c(Rank.SIX, Suit.CLUBS), c(Rank.SEVEN, Suit.HEARTS)),
             hand(c(Rank.SEVEN, Suit.CLUBS), c(Rank.NINE, Suit.HEARTS)),
             hand(c(Rank.NINE, Suit.CLUBS), c(Rank.TEN, Suit.HEARTS))],
            leader=0)
        engine.step((engine.hands[0][0],))    # 0 plays the 2 and finishes
        engine.step(PASS)                     # 1
        engine.step(PASS)                     # 2
        events = engine.step(PASS)            # 3 - trick over, winner is out
        assert ('trick_won', 0) in events
        assert engine.current == 1            # next player with cards leads

    def test_no_steps_after_round_over(self):
        engine = TrickEngine(
            [hand(c(Rank.FIVE, Suit.CLUBS)),
             hand(c(Rank.SIX, Suit.CLUBS)),
             hand(c(Rank.SEVEN, Suit.CLUBS)),
             hand(c(Rank.NINE, Suit.CLUBS))],
            leader=0)
        engine.step((engine.hands[0][0],))
        engine.step((engine.hands[1][0],))
        engine.step((engine.hands[2][0],))
        assert engine.round_over()
        with pytest.raises(ValueError):
            engine.step(PASS)
