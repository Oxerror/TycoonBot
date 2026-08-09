"""The caching reader must return exactly what the live reader would,
while touching the real recognition only once per frame."""

from GameLogic.Card import Card, Rank, Suit
from FrameReader import CachedFrameReader


class FakeReader:
    """Stands in for FrameReader and counts every real read."""

    def __init__(self):
        self.calls = 0
        self.bar_value = {rank: 0 for rank in Rank}
        self.bar_value[Rank.ACE] = 3

    def bar(self, frame):
        self.calls += 1
        return dict(self.bar_value)

    def counters(self, frame):
        self.calls += 1
        return ({'left': 5, 'middle': None, 'right': 3, 'player': 14},
                ['middle'], 'player', ['right'])

    def field(self, play_field):
        self.calls += 1
        return False, [Card(Rank.EIGHT, Suit.HEARTS), Card(Rank.JOKER)]

    def hand(self, hand_crop):
        self.calls += 1
        return ['raw-detections'], [Card(Rank.WONDER), Card(Rank.TWO, Suit.CLUBS)]

    def revolution(self, frame):
        self.calls += 1
        return True


def read_everything(reader):
    return (reader.bar(None), reader.counters(None), reader.field(None),
            reader.hand(None), reader.revolution(None))


def test_cache_round_trips_all_readings(tmp_path):
    frame = tmp_path / 'frame.png'
    frame.touch()

    fake = FakeReader()
    reader = CachedFrameReader(fake)
    reader.begin_frame(frame)
    first = read_everything(reader)
    assert fake.calls == 5

    # Same frame again, even across a fresh reader: no real reads.
    reader = CachedFrameReader(fake)
    reader.begin_frame(frame)
    second = read_everything(reader)
    assert fake.calls == 5

    assert second[0] == first[0]                      # bar, Rank keys
    assert list(second[0]) == list(first[0])
    assert second[1] == first[1]                      # counters tuple
    assert second[2][0] is False                      # banner flag
    assert [(c.rank, c.suit) for c in second[2][1]] == [
        (Rank.EIGHT, Suit.HEARTS), (Rank.JOKER, None)]
    assert second[3][0] == []                         # detections dropped
    assert [(c.rank, c.suit) for c in second[3][1]] == [
        (Rank.WONDER, None), (Rank.TWO, Suit.CLUBS)]
    assert second[4] is True


def test_none_bar_is_cached(tmp_path):
    frame = tmp_path / 'frame.png'
    frame.touch()

    class NoBarReader(FakeReader):
        def bar(self, frame):
            self.calls += 1
            return None

    fake = NoBarReader()
    reader = CachedFrameReader(fake)
    reader.begin_frame(frame)
    assert reader.bar(None) is None
    assert reader.bar(None) is None
    assert fake.calls == 1


def test_new_frame_reads_again(tmp_path):
    fake = FakeReader()
    reader = CachedFrameReader(fake)
    for name in ('a.png', 'b.png'):
        frame = tmp_path / name
        frame.touch()
        reader.begin_frame(frame)
        reader.bar(None)
    assert fake.calls == 2


def test_version_bump_invalidates(tmp_path):
    frame = tmp_path / 'frame.png'
    frame.touch()

    fake = FakeReader()
    reader = CachedFrameReader(fake)
    reader.begin_frame(frame)
    reader.bar(None)
    assert fake.calls == 1

    sidecar = tmp_path / 'frame.png.readings.json'
    assert sidecar.exists()
    sidecar.write_text(sidecar.read_text().replace('"version": 1',
                                                   '"version": 0'))
    reader = CachedFrameReader(fake)
    reader.begin_frame(frame)
    reader.bar(None)
    assert fake.calls == 2


def test_no_begin_frame_means_no_caching():
    fake = FakeReader()
    reader = CachedFrameReader(fake)
    reader.bar(None)
    reader.bar(None)
    assert fake.calls == 2
