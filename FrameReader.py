"""Per-frame recognition bundled behind one object, so it can be cached.

Recognition is ~99% of replay time and deterministic per frame, while
the game-following logic on top changes constantly during development.
Splitting the two lets Replay run with a CachedFrameReader: the first
pass over a capture stores every reading in a JSON sidecar next to the
frame, and every later replay skips template matching entirely —
seconds instead of minutes. Delete the *.readings.json sidecars (or
bump CACHE_VERSION when readings change shape) to force fresh reads.

The live loop uses the plain FrameReader; nothing is cached there.

One caveat: the recognizer warms a per-template scale cache as it
goes, so a borderline detection (a card clipped at the fan edge) can
differ between runs that processed different frames beforehand —
including between a cached reading and a fresh sequential replay.
The tracking layer absorbs those single-card wobbles; delete the
sidecars if a recognition change needs a truly fresh baseline.
"""

import json
from pathlib import Path

from GameLogic.Card import Card, Rank, Suit
from GameLogic.HandReader import detections_to_cards
from CardsLeftReader import read_cards_left_detailed
from ImageRecognition import (banner_visible, get_recognizer,
                              read_hand_detections, read_play_field,
                              read_revolution_indicator)
from StatusBarReader import read_status_bar

CACHE_VERSION = 1


class FrameReader:
    """The live reader: every call runs the real recognition."""

    def __init__(self):
        self.recognizer = get_recognizer()

    def bar(self, frame):
        """Status bar rank counts, or None when the bar is unreadable."""
        return read_status_bar(frame)

    def counters(self, frame):
        """(counts, unknown, active, passed) from the Cards Left
        bubbles."""
        return read_cards_left_detailed(frame)

    def field(self, play_field):
        """(banner_visible, trick cards); the trick is empty while an
        event banner covers the table."""
        if banner_visible(play_field):
            return True, []
        return False, read_play_field(play_field)

    def hand(self, hand_crop):
        """(raw detections, Cards) of the own hand."""
        detections = read_hand_detections(hand_crop)
        return detections, detections_to_cards(detections)

    def revolution(self, frame):
        """True while the Flip Strength badge shows."""
        return read_revolution_indicator(frame)


def _encode_card(card):
    return [card.rank.name, card.suit.name if card.suit else None]


def _decode_card(data):
    rank, suit = data
    return Card(Rank[rank], Suit[suit] if suit else None)


class CachedFrameReader:
    """Read-through cache around a FrameReader.

    Call begin_frame(path) before each frame; readings are stored in
    `<frame>.readings.json` beside it. Raw hand detections are not
    cached (they only feed the live overlay), so a cache hit returns
    an empty detections list alongside the cards.
    """

    def __init__(self, inner=None):
        self._inner = inner
        self.sidecar = None
        self.cache = {}

    @property
    def inner(self):
        """Created on first miss: a fully cached replay never pays for
        loading the recognizer."""
        if self._inner is None:
            self._inner = FrameReader()
        return self._inner

    def begin_frame(self, frame_path):
        path = Path(frame_path)
        self.sidecar = path.with_name(path.name + '.readings.json')
        self.cache = {}
        if self.sidecar.exists():
            data = json.loads(self.sidecar.read_text())
            if data.get('version') == CACHE_VERSION:
                self.cache = data

    def _get(self, kind, compute, encode=lambda v: v, decode=lambda v: v,
             valid=lambda stored: True):
        if kind in self.cache and valid(self.cache[kind]):
            return decode(self.cache[kind])
        value = compute()
        if self.sidecar is not None:
            self.cache[kind] = encode(value)
            self.cache['version'] = CACHE_VERSION
            self.sidecar.write_text(json.dumps(self.cache))
        return value

    def bar(self, frame):
        return self._get(
            'bar', lambda: self.inner.bar(frame),
            encode=lambda bar: (None if bar is None
                                else {r.name: n for r, n in bar.items()}),
            decode=lambda bar: (None if bar is None
                                else {Rank[r]: n for r, n in bar.items()}))

    def counters(self, frame):
        # 'counters_v2': entries recorded before the turn-button
        # detector under the old key are stale; the counter read is
        # cheap, so they simply re-run. `valid` likewise rejects
        # entries from before the pass marker existed.
        return self._get(
            'counters_v2', lambda: self.inner.counters(frame),
            encode=lambda value: [value[0], list(value[1]), value[2],
                                  list(value[3])],
            decode=lambda value: (value[0], value[1], value[2], value[3]),
            valid=lambda stored: len(stored) == 4)

    def field(self, play_field):
        # 'field_v3': suit-covered ranks now count as suitless cards
        # (with the mirrored bottom-corner glyphs vetoed), so field
        # readings cached by earlier conversions are stale.
        return self._get(
            'field_v3', lambda: self.inner.field(play_field),
            encode=lambda value: [value[0], [_encode_card(c) for c in value[1]]],
            decode=lambda value: (value[0], [_decode_card(c) for c in value[1]]))

    def hand(self, hand_crop):
        # 'hand_v2': the hand read gained a second pass for greyed-out
        # card glyphs and the hand crop reaches higher (lifted cards),
        # so readings cached under the old key are stale.
        return self._get(
            'hand_v2', lambda: self.inner.hand(hand_crop),
            encode=lambda value: [_encode_card(c) for c in value[1]],
            decode=lambda cards: ([], [_decode_card(c) for c in cards]))

    def revolution(self, frame):
        return self._get('revolution', lambda: self.inner.revolution(frame))
