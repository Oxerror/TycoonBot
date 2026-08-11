"""The gamepad backends against a fake vgamepad module.

The real vgamepad needs the ViGEmBus kernel driver, so these tests
inject a stand-in and check what the backends would do to it: the
button mapping, the press -> update -> release -> update choreography,
and the DS4's hat-switch D-pad. The real-device path is the manual
smoke test (python VirtualGamepad.py, watched in joy.cpl).
"""

import importlib.util
from types import SimpleNamespace

import pytest

from InputPlanner import (CIRCLE, CROSS, DPAD_LEFT, DPAD_RIGHT, SQUARE,
                          TRIANGLE)
from VirtualGamepad import DS4Backend, XboxBackend


class FakePad:
    def __init__(self):
        self.calls = []

    def press_button(self, button):
        self.calls.append(('press', button))

    def release_button(self, button):
        self.calls.append(('release', button))

    def directional_pad(self, direction):
        self.calls.append(('dpad', direction))

    def update(self):
        self.calls.append(('update',))


def fake_vg():
    """Enum values are plain strings so assertions read naturally."""
    return SimpleNamespace(
        VX360Gamepad=FakePad,
        VDS4Gamepad=FakePad,
        XUSB_BUTTON=SimpleNamespace(
            XUSB_GAMEPAD_A='A', XUSB_GAMEPAD_B='B', XUSB_GAMEPAD_X='X',
            XUSB_GAMEPAD_Y='Y', XUSB_GAMEPAD_DPAD_LEFT='LEFT',
            XUSB_GAMEPAD_DPAD_RIGHT='RIGHT'),
        DS4_BUTTONS=SimpleNamespace(
            DS4_BUTTON_CROSS='CROSS', DS4_BUTTON_CIRCLE='CIRCLE',
            DS4_BUTTON_SQUARE='SQUARE', DS4_BUTTON_TRIANGLE='TRIANGLE'),
        DS4_DPAD_DIRECTIONS=SimpleNamespace(
            DS4_BUTTON_DPAD_WEST='WEST', DS4_BUTTON_DPAD_EAST='EAST',
            DS4_BUTTON_DPAD_NONE='NONE'),
    )


def xbox():
    return XboxBackend(vg=fake_vg(), press_seconds=0, gap_seconds=0)


def ds4():
    return DS4Backend(vg=fake_vg(), press_seconds=0, gap_seconds=0)


class TestXbox:
    def test_face_buttons_land_on_xinput_positions(self):
        backend = xbox()
        for name, expected in ((CROSS, 'A'), (CIRCLE, 'B'),
                               (SQUARE, 'X'), (TRIANGLE, 'Y')):
            backend.pad.calls.clear()
            backend.press(name)
            assert backend.pad.calls == [('press', expected), ('update',),
                                         ('release', expected), ('update',)]

    def test_dpad_is_ordinary_buttons(self):
        backend = xbox()
        backend.press(DPAD_LEFT)
        assert backend.pad.calls[0] == ('press', 'LEFT')

    def test_sequence_presses_in_order(self):
        backend = xbox()
        backend.press_sequence([DPAD_RIGHT, CROSS, TRIANGLE])
        pressed = [call[1] for call in backend.pad.calls
                   if call[0] == 'press']
        assert pressed == ['RIGHT', 'A', 'Y']

    def test_unknown_button_refuses(self):
        with pytest.raises(KeyError):
            xbox().press('L3')


class TestDS4:
    def test_face_buttons_keep_their_names(self):
        backend = ds4()
        backend.press(TRIANGLE)
        assert backend.pad.calls == [('press', 'TRIANGLE'), ('update',),
                                     ('release', 'TRIANGLE'), ('update',)]

    def test_dpad_sets_and_clears_the_hat(self):
        backend = ds4()
        backend.press(DPAD_LEFT)
        assert backend.pad.calls == [('dpad', 'WEST'), ('update',),
                                     ('dpad', 'NONE'), ('update',)]

    def test_planner_vocabulary_is_fully_mapped(self):
        backend = ds4()
        backend.press_sequence([DPAD_LEFT, DPAD_RIGHT, CROSS, CIRCLE,
                                SQUARE, TRIANGLE])
        assert len([c for c in backend.pad.calls if c == ('update',)]) == 12


def test_missing_vgamepad_explains_itself():
    if importlib.util.find_spec('vgamepad') is not None:
        pytest.skip("vgamepad is installed here")
    with pytest.raises(ImportError, match='ViGEmBus'):
        XboxBackend()
