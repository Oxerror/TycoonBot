"""Virtual gamepad backends: button names in, real pad presses out.

Built on vgamepad, which drives the ViGEmBus kernel driver
(`pip install vgamepad` offers to install it). Two pads are available:

- XboxBackend emulates an Xbox 360 pad (XInput) — the default, since
  virtually everything on PC accepts it.
- DS4Backend emulates a DualShock 4 — for PS Remote Play, which only
  listens to PlayStation controllers.

Smoke test WITHOUT the game running:

    python VirtualGamepad.py [--ds4]

presses every mapped button once per second while you watch the pad
respond in Windows' controller panel (Win+R -> joy.cpl -> select the
controller -> Properties) or at https://hardwaretester.com/gamepad.
If a "Controller (XBOX 360 For Windows)" appears there and the button
lights follow the console output, the input path works end to end.

Wired into the live loop only on request: `python VideoStream.py
--act {xbox,ds4}` builds an act-mode executor around one of these
backends; without the flag the loop stays suggest-only. The timings
below were calibrated in the supervised session of 2026-08-16: at the
old 0.05/0.08 placeholders PS Remote Play dropped face-button presses
outright, at 0.12/0.35 every press landed.
"""

import time

from InputPlanner import (CIRCLE, CROSS, DPAD_LEFT, DPAD_RIGHT, OPTIONS,
                          SQUARE, TRIANGLE)

PRESS_SECONDS = 0.12
GAP_SECONDS = 0.35


class GamepadBackend:
    """Shared press/release choreography; subclasses map the buttons.

    The vgamepad module is only imported when a backend is actually
    constructed (it needs the ViGEmBus driver); tests inject a fake
    `vg` instead.
    """

    def __init__(self, vg=None, press_seconds=PRESS_SECONDS,
                 gap_seconds=GAP_SECONDS):
        if vg is None:
            try:
                import vgamepad as vg
            except ImportError as error:
                raise ImportError(
                    "vgamepad is not installed - pip install vgamepad "
                    "(its setup offers the required ViGEmBus driver)"
                ) from error
        self.vg = vg
        self.press_seconds = press_seconds
        self.gap_seconds = gap_seconds
        self.pad = self._make_pad()

    def press_sequence(self, plan):
        """Press the planned buttons one by one, in order."""
        for button in plan:
            self.press(button)

    def press(self, button):
        self._press(button)
        self.pad.update()
        time.sleep(self.press_seconds)
        self._release(button)
        self.pad.update()
        time.sleep(self.gap_seconds)


class XboxBackend(GamepadBackend):
    """Xbox 360 pad; PlayStation face buttons land on their XInput
    positions (CROSS=A, CIRCLE=B, SQUARE=X, TRIANGLE=Y, OPTIONS=START)."""

    def __init__(self, vg=None, **timings):
        super().__init__(vg, **timings)
        buttons = self.vg.XUSB_BUTTON
        self._buttons = {
            CROSS: buttons.XUSB_GAMEPAD_A,
            CIRCLE: buttons.XUSB_GAMEPAD_B,
            SQUARE: buttons.XUSB_GAMEPAD_X,
            TRIANGLE: buttons.XUSB_GAMEPAD_Y,
            OPTIONS: buttons.XUSB_GAMEPAD_START,
            DPAD_LEFT: buttons.XUSB_GAMEPAD_DPAD_LEFT,
            DPAD_RIGHT: buttons.XUSB_GAMEPAD_DPAD_RIGHT,
        }

    def _make_pad(self):
        return self.vg.VX360Gamepad()

    def _press(self, button):
        self.pad.press_button(button=self._buttons[button])

    def _release(self, button):
        self.pad.release_button(button=self._buttons[button])


class DS4Backend(GamepadBackend):
    """DualShock 4 pad. The D-pad is not a set of buttons on a DS4 but
    one hat direction, set and cleared as a whole."""

    def __init__(self, vg=None, **timings):
        super().__init__(vg, **timings)
        buttons = self.vg.DS4_BUTTONS
        self._buttons = {
            CROSS: buttons.DS4_BUTTON_CROSS,
            CIRCLE: buttons.DS4_BUTTON_CIRCLE,
            SQUARE: buttons.DS4_BUTTON_SQUARE,
            TRIANGLE: buttons.DS4_BUTTON_TRIANGLE,
            OPTIONS: buttons.DS4_BUTTON_OPTIONS,
        }
        directions = self.vg.DS4_DPAD_DIRECTIONS
        self._dpad = {
            DPAD_LEFT: directions.DS4_BUTTON_DPAD_WEST,
            DPAD_RIGHT: directions.DS4_BUTTON_DPAD_EAST,
        }
        self._dpad_released = directions.DS4_BUTTON_DPAD_NONE

    def _make_pad(self):
        return self.vg.VDS4Gamepad()

    def _press(self, button):
        if button in self._dpad:
            self.pad.directional_pad(direction=self._dpad[button])
        else:
            self.pad.press_button(button=self._buttons[button])

    def _release(self, button):
        if button in self._dpad:
            self.pad.directional_pad(direction=self._dpad_released)
        else:
            self.pad.release_button(button=self._buttons[button])


if __name__ == '__main__':
    import sys

    kind = 'DS4' if '--ds4' in sys.argv else 'Xbox'
    backend = (DS4Backend() if kind == 'DS4' else XboxBackend())
    print(f"{kind} pad created - watch it in joy.cpl (Properties) or "
          "https://hardwaretester.com/gamepad")
    time.sleep(2)  # give the tester a moment to enumerate the new pad
    for name in (DPAD_LEFT, DPAD_RIGHT, CROSS, CIRCLE, SQUARE, TRIANGLE,
                 OPTIONS):
        print(f"  pressing {name}")
        backend.press(name)
        time.sleep(1)
    print("Done - every button above should have flashed exactly once.")
