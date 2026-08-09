"""Play whole rounds of Tycoon offline.

Glues the TrickEngine to pluggable player policies so recommender
ideas can be measured against each other long before the bot ever
touches the real game: deal seeded hands, let four policies fight,
read off the finish order.

A policy is a callable (hand, trick, revolution, unseen) -> move,
where unseen counts the cards hidden from that player (everything not
in their own hand and not yet played) — exactly what GameState tracks
for the live player, so `recommend` drops in unchanged.
"""

from collections import Counter

from GameLogic.Card import Card, Rank, Suit
from GameLogic.GameState import FULL_DECK
from GameLogic.Recommender import recommend
from GameLogic.Rules import PASS, legal_moves
from GameLogic.TrickEngine import TrickEngine

PLAYER_COUNT = 4


def build_deck():
    """The full 56-card Tycoon deck."""
    deck = [Card(rank, suit)
            for rank in Rank if rank not in (Rank.JOKER, Rank.WONDER)
            for suit in Suit]
    deck += [Card(Rank.JOKER), Card(Rank.JOKER),
             Card(Rank.WONDER), Card(Rank.WONDER)]
    return deck


def deal(rng):
    """Shuffle and deal four hands of 14."""
    deck = build_deck()
    rng.shuffle(deck)
    share = len(deck) // PLAYER_COUNT
    return [deck[i * share:(i + 1) * share] for i in range(PLAYER_COUNT)]


def first_leader(hands):
    """The holder of the 3 of Diamonds leads round one."""
    for player, hand in enumerate(hands):
        if any(card.rank == Rank.THREE and card.suit == Suit.DIAMONDS
               for card in hand):
            return player
    raise ValueError("No hand holds the 3 of Diamonds")


def recommender_policy(hand, trick, revolution, unseen):
    return recommend(hand, trick, revolution, unseen)


def random_policy(rng):
    """Uniform pick among the legal sets, plus PASS when following."""
    def policy(hand, trick, revolution, unseen):
        options = legal_moves(hand, trick, revolution)
        if trick:
            options = options + [PASS]
        if not options:
            return PASS
        return options[rng.randrange(len(options))]
    return policy


def play_round(hands, leader, policies, on_event=None):
    """
    Run one round to completion.

    Args:
        hands: the dealt hands (consumed by the engine's own copies)
        leader: index of the player leading the first trick
        policies: one policy per player
        on_event: optional callable receiving every engine event

    Returns:
        The finish order: Tycoon first, Beggar last.
    """
    engine = TrickEngine(hands, leader)
    played = Counter()

    while not engine.round_over():
        player = engine.current
        hand = engine.hands[player]
        own = Counter(card.rank for card in hand)
        unseen = {rank: FULL_DECK[rank] - played[rank] - own[rank]
                  for rank in Rank}

        move = policies[player](hand, engine.trick, engine.revolution, unseen)
        for event in engine.step(move if move else PASS):
            if event[0] == 'play':
                played.update(card.rank for card in event[2])
            if on_event is not None:
                on_event(event)

    return engine.ranking()
