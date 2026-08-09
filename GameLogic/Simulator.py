"""Play whole rounds of Tycoon offline.

Glues the TrickEngine to pluggable player policies so recommender
ideas can be measured against each other long before the bot ever
touches the real game: deal seeded hands, let four policies fight,
read off the finish order.

A policy is a callable Observation -> move. The Observation carries
everything a player may legally know: the own hand, the table, the
unseen counts (everything not in the own hand and not yet played —
exactly what GameState tracks for the live player, so `recommend`
drops in unchanged), everyone's hand sizes, and the trick bookkeeping
(who laid down the current set, who has passed).
"""

from collections import Counter, namedtuple

from GameLogic.Card import Card, Rank, Suit
from GameLogic.GameState import FULL_DECK
from GameLogic.Recommender import recommend
from GameLogic.Rules import PASS, legal_moves
from GameLogic.TrickEngine import TrickEngine

PLAYER_COUNT = 4

Observation = namedtuple('Observation', [
    'seat',         # own player index
    'hand',         # own cards
    'trick',        # the set on the table, () when leading
    'revolution',   # True while a revolution is active
    'unseen',       # {Rank: count} hidden from this player
    'counts',       # cards left per player, by seat
    'passed',       # seats locked out of the current trick
    'last_player',  # who laid down the current set, None when leading
])


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


def recommender_policy(obs):
    return recommend(obs.hand, obs.trick, obs.revolution, obs.unseen)


def random_policy(rng):
    """Uniform pick among the legal sets, plus PASS when following."""
    def policy(obs):
        options = legal_moves(obs.hand, obs.trick, obs.revolution)
        if obs.trick:
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
        obs = Observation(seat=player,
                          hand=tuple(hand),
                          trick=engine.trick,
                          revolution=engine.revolution,
                          unseen=unseen,
                          counts=tuple(len(h) for h in engine.hands),
                          passed=frozenset(engine.passed),
                          last_player=engine.last_player)

        move = policies[player](obs)
        for event in engine.step(move if move else PASS):
            if event[0] == 'play':
                played.update(card.rank for card in event[2])
            if on_event is not None:
                on_event(event)

    return engine.ranking()
