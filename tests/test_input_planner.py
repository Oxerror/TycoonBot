"""The move->button planner on synthetic hands.

Every expectation here encodes the UI_ASSUMPTIONS defaults (cursor at
the leftmost card, wrap-around enabled, CROSS selects, TRIANGLE
confirms, SQUARE passes); when a live session corrects an assumption,
the affected expectations change with it.
"""

import pytest

from GameLogic.Card import Card, Rank, Suit
from GameLogic.Rules import PASS
from InputPlanner import (CROSS, DPAD_LEFT, DPAD_RIGHT, SQUARE, TRIANGLE,
                          merge_into_fan, plan_move)


def c(rank, suit=None):
    return Card(rank, suit)


FAN = [c(Rank.THREE, Suit.DIAMONDS), c(Rank.FIVE, Suit.CLUBS),
       c(Rank.FIVE, Suit.HEARTS), c(Rank.KING, Suit.SPADES),
       c(Rank.TWO, Suit.HEARTS)]


class TestSelecting:
    def test_card_under_the_cursor_needs_no_steps(self):
        plan = plan_move(FAN, (FAN[0],))
        assert plan == [CROSS, TRIANGLE]

    def test_steps_right_to_reach_a_card(self):
        plan = plan_move(FAN, (FAN[2],))
        assert plan == [DPAD_RIGHT, DPAD_RIGHT, CROSS, TRIANGLE]

    def test_pair_is_selected_left_to_right(self):
        plan = plan_move(FAN, (FAN[2], FAN[1]))
        assert plan == [DPAD_RIGHT, CROSS, DPAD_RIGHT, CROSS, TRIANGLE]

    def test_selection_order_ignores_move_order(self):
        assert plan_move(FAN, (FAN[3], FAN[0])) == plan_move(FAN, (FAN[0], FAN[3]))

    def test_single_card_fan(self):
        assert plan_move([c(Rank.ACE, Suit.CLUBS)],
                         (c(Rank.ACE, Suit.CLUBS),)) == [CROSS, TRIANGLE]


class TestCursorWrap:
    def test_wraps_to_the_last_card_when_shorter(self):
        plan = plan_move(FAN, (FAN[4],))
        assert plan == [DPAD_LEFT, CROSS, TRIANGLE]

    def test_prefers_right_on_equal_distance(self):
        fan = FAN[:4]
        plan = plan_move(fan, (fan[2],))
        assert plan == [DPAD_RIGHT, DPAD_RIGHT, CROSS, TRIANGLE]

    def test_without_wrap_steps_all_the_way(self):
        plan = plan_move(FAN, (FAN[4],), ui={'cursor_wraps': False})
        assert plan == [DPAD_RIGHT] * 4 + [CROSS, TRIANGLE]


class TestPassing:
    def test_pass_is_one_button(self):
        assert plan_move(FAN, PASS) == [SQUARE]

    def test_pass_with_confirm_flow(self):
        plan = plan_move(FAN, PASS, ui={'pass_needs_confirm': True})
        assert plan == [SQUARE, TRIANGLE]

    def test_pass_needs_no_fan(self):
        assert plan_move([], PASS) == [SQUARE]


class TestCardMatching:
    def test_suit_disambiguates_equal_ranks(self):
        """Card.__eq__ is rank-only; the planner must not select the
        3 of Diamonds when the move needs the 3 of Spades."""
        fan = [c(Rank.THREE, Suit.DIAMONDS), c(Rank.THREE, Suit.SPADES)]
        plan = plan_move(fan, (c(Rank.THREE, Suit.SPADES),))
        assert plan == [DPAD_RIGHT, CROSS, TRIANGLE]

    def test_duplicate_jokers_take_distinct_slots(self):
        fan = [c(Rank.FOUR, Suit.CLUBS), c(Rank.JOKER), c(Rank.JOKER)]
        plan = plan_move(fan, (c(Rank.JOKER), c(Rank.JOKER)))
        assert plan == [DPAD_RIGHT, CROSS, DPAD_RIGHT, CROSS, TRIANGLE]

    def test_suitless_placeholder_matches_suited_fan_card(self):
        """The tracker holds bar-recovered cards without a suit; they
        must still find their slot in the recognized fan."""
        plan = plan_move(FAN, (c(Rank.KING),))
        assert plan == [DPAD_LEFT, DPAD_LEFT, CROSS, TRIANGLE]

    def test_suited_move_card_matches_placeholder_slot(self):
        fan = [c(Rank.THREE), c(Rank.KING, Suit.SPADES)]
        plan = plan_move(fan, (c(Rank.THREE, Suit.HEARTS),))
        assert plan == [CROSS, TRIANGLE]

    def test_missing_card_refuses_to_guess(self):
        with pytest.raises(ValueError):
            plan_move(FAN, (c(Rank.ACE, Suit.SPADES),))

    def test_more_copies_than_the_fan_holds_refuses(self):
        with pytest.raises(ValueError):
            plan_move(FAN, (c(Rank.KING, Suit.SPADES), c(Rank.KING, Suit.HEARTS)))

    def test_empty_fan_cannot_host_a_play(self):
        with pytest.raises(ValueError):
            plan_move([], (c(Rank.THREE, Suit.DIAMONDS),))


class TestUiOverrides:
    def test_custom_buttons(self):
        plan = plan_move(FAN, (FAN[0],),
                         ui={'select_button': 'L1', 'confirm_button': 'R1'})
        assert plan == ['L1', 'R1']

    def test_cursor_start_elsewhere(self):
        plan = plan_move(FAN, (FAN[3],), ui={'cursor_start': 4})
        assert plan == [DPAD_LEFT, CROSS, TRIANGLE]


class TestMergeIntoFan:
    def test_identical_reading_changes_nothing(self):
        assert merge_into_fan(FAN, list(FAN)) == FAN

    def test_clipped_low_card_slots_in_at_the_left(self):
        fan = merge_into_fan(FAN[1:], [c(Rank.THREE)] + FAN[1:])
        assert fan[0] == c(Rank.THREE)
        assert fan[0].suit is None
        assert fan[1:] == FAN[1:]

    def test_clipped_joker_slots_in_at_the_right(self):
        fan = merge_into_fan(FAN, list(FAN) + [c(Rank.JOKER)])
        assert fan[:-1] == FAN
        assert fan[-1].rank == Rank.JOKER

    def test_wonder_belongs_leftmost(self):
        fan = merge_into_fan(FAN, [c(Rank.WONDER)] + list(FAN))
        assert fan[0].rank == Rank.WONDER

    def test_placeholder_already_visible_is_not_duplicated(self):
        """A suitless tracked KING is the recognized KING_SPADES, not
        an extra card."""
        hand = [c(Rank.KING)] + [card for card in FAN if card.rank != Rank.KING]
        assert merge_into_fan(FAN, hand) == FAN

    def test_misread_extra_card_keeps_its_slot(self):
        """A slot whose label the tracker rules out is still a slot the
        cursor steps across."""
        hand = [card for card in FAN if card.rank != Rank.FIVE]
        assert merge_into_fan(FAN, hand) == FAN

    def test_merged_fan_is_plannable(self):
        """End to end: a bar-recovered card is selectable after the merge."""
        recognized = FAN[1:]
        hand = [c(Rank.THREE)] + FAN[1:]
        plan = plan_move(merge_into_fan(recognized, hand), (c(Rank.THREE),))
        assert plan == [CROSS, TRIANGLE]
