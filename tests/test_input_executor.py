import pytest

from InputExecutor import InputExecutor


class FakeBackend:
    def __init__(self):
        self.sent = []

    def press_sequence(self, plan):
        self.sent.append(list(plan))


class TestSuggestMode:
    def test_is_the_default(self):
        assert InputExecutor().mode == 'suggest'

    def test_records_but_never_presses(self):
        backend = FakeBackend()
        executor = InputExecutor(backend=backend)
        assert executor.execute(['CROSS', 'TRIANGLE']) is False
        assert backend.sent == []
        assert executor.history == [['CROSS', 'TRIANGLE']]


class TestActMode:
    def test_presses_through_the_backend(self):
        backend = FakeBackend()
        executor = InputExecutor(backend=backend, mode='act')
        assert executor.execute(['SQUARE']) is True
        assert backend.sent == [['SQUARE']]
        assert executor.history == [['SQUARE']]

    def test_requires_a_backend(self):
        with pytest.raises(ValueError):
            InputExecutor(mode='act')


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        InputExecutor(mode='autopilot')
