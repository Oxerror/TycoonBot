"""The --act flag is the only door to an act-mode executor.

buildExecutor is the single constructor of act mode in the codebase:
without the flag the live loop must come up suggest-only, with it the
requested pad backend must actually be behind the executor.
"""

import pytest

import VirtualGamepad
from VideoStream import buildExecutor


def test_default_is_suggest_only():
    executor = buildExecutor(None)
    assert executor.mode == 'suggest'
    assert executor.backend is None


@pytest.mark.parametrize('kind, attribute', [('xbox', 'XboxBackend'),
                                             ('ds4', 'DS4Backend')])
def test_act_builds_the_requested_pad(monkeypatch, kind, attribute):
    sentinel = object()
    monkeypatch.setattr(VirtualGamepad, attribute, lambda: sentinel)
    executor = buildExecutor(kind)
    assert executor.mode == 'act'
    assert executor.backend is sentinel
