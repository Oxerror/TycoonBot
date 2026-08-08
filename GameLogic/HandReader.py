"""Convert recognizer detections into Card objects.

The recognizer reports rank glyphs (e.g. 'King') and suit glyphs
(e.g. 'Heart') independently, each with a bounding box. On a card the
suit symbol is drawn directly below the rank, so a rank is paired with
the nearest suit glyph whose center lies below its own.
"""

from GameLogic.Card import Card, Rank, Suit

RANK_BY_NAME = {
    '2': Rank.TWO,
    '3': Rank.THREE,
    '4': Rank.FOUR,
    '5': Rank.FIVE,
    '6': Rank.SIX,
    '7': Rank.SEVEN,
    '8': Rank.EIGHT,
    '9': Rank.NINE,
    '10': Rank.TEN,
    'Jack': Rank.JACK,
    'Queen': Rank.QUEEN,
    'King': Rank.KING,
    'Ace': Rank.ACE,
    'Joker': Rank.JOKER,
    'Wonder': Rank.WONDER,
}

SUIT_BY_NAME = {
    'Spade': Suit.SPADES,
    'Heart': Suit.HEARTS,
    'Diamond': Suit.DIAMONDS,
    'Cross': Suit.CLUBS,
}


def _center(detection):
    x, y, w, h = detection['location']
    return (x + w / 2, y + h / 2)


def _display_key(card):
    """Position of a card in the game's hand display order.

    The game always shows the hand sorted: Wonder leftmost, then game
    order (3 lowest up to Ace, then 2), Joker rightmost. Rank values
    already encode 3..2..Joker; only Wonder displays out of rank order.
    """
    if card.rank == Rank.WONDER:
        return 0
    return card.rank.value


def hand_is_ordered(cards):
    """
    True when a hand reading respects the game's display order.

    The game keeps the hand sorted, so a left-to-right reading that is
    out of order proves at least one card was misrecognized.
    """
    keys = [_display_key(card) for card in cards]
    return all(a <= b for a, b in zip(keys, keys[1:]))


def detections_to_cards(detections):
    """
    Pair rank and suit detections into Card objects.

    Args:
        detections: List of dicts from CardRecognizer.template_match,
            each with 'name', 'confidence' and 'location' (x, y, w, h).

    Returns:
        List of Cards sorted left to right by screen position. Rank
        glyphs that cannot be paired with a suit are skipped, except
        Joker and Wonder which have no suit.
    """
    ranks = [d for d in detections if d['name'] in RANK_BY_NAME]
    suits = [d for d in detections if d['name'] in SUIT_BY_NAME]

    ranks.sort(key=lambda d: d['location'][0])

    cards = []
    available_suits = list(suits)

    for rank_det in ranks:
        rank = RANK_BY_NAME[rank_det['name']]

        if rank in (Rank.JOKER, Rank.WONDER):
            cards.append(Card(rank))
            continue

        rank_cx, rank_cy = _center(rank_det)
        rank_h = rank_det['location'][3]

        candidates = []
        for suit_det in available_suits:
            suit_cx, suit_cy = _center(suit_det)

            # The suit symbol sits below the rank on the same card. The
            # horizontal window must stay well below the card spacing,
            # otherwise a neighbor card's (more confident) suit can win
            # and set off a chain of off-by-one pairings.
            if suit_cy <= rank_cy:
                continue
            if abs(suit_cx - rank_cx) > rank_h * 0.45:
                continue

            dist = (suit_cx - rank_cx) ** 2 + (suit_cy - rank_cy) ** 2
            candidates.append((dist, suit_det))

        if not candidates:
            continue

        # Card artwork produces occasional low-confidence suit matches that
        # can be closer to the rank than the real suit symbol. Real suit
        # symbols score very high, so prefer confident candidates.
        confident = [c for c in candidates if c[1]['confidence'] >= 0.85]
        best_suit = min(confident or candidates, key=lambda c: c[0])[1]

        available_suits.remove(best_suit)
        cards.append(Card(rank, SUIT_BY_NAME[best_suit['name']]))

    return cards
