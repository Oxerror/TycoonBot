"""Play whole rounds — and whole games — of Tycoon offline.

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

`play_game` chains rounds into a full game: after round one the
Beggar leads, and the round starts with the card exchange — the
Beggar's two best cards go to the Tycoon, the Poor's best to the
Rich, and the receivers hand back freely chosen cards (weakest by
default; a policy object may override the choice with an
`exchange(hand, count)` method returning cards from that hand).
"""

from collections import Counter, namedtuple

from GameLogic.Card import Card, Rank, Suit
from GameLogic.GameState import FULL_DECK
from GameLogic.Recommender import recommend
from GameLogic.Rules import PASS, effective_strength, legal_moves
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
    # Match context (defaults: a bare first round). `roles` is the
    # previous round's finish place per seat (0 Tycoon .. 3 Beggar),
    # None when there is no previous round — the exchange shapes the
    # hands by role, so policies deserve to know who is stacked.
    'roles',
    'round_index',  # 0-based round number within the match
], defaults=(None, 0))


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


def play_round(hands, leader, policies, on_event=None, roles=None,
               round_index=0):
    """
    Run one round to completion.

    Args:
        hands: the dealt hands (consumed by the engine's own copies)
        leader: index of the player leading the first trick
        policies: one policy per player
        on_event: optional callable receiving every engine event
        roles: previous round's finish place per seat, None in a
            bare first round (carried into every Observation)
        round_index: 0-based round number within the match

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
                          last_player=engine.last_player,
                          roles=roles,
                          round_index=round_index)

        move = policies[player](obs)
        for event in engine.step(move if move else PASS):
            if event[0] == 'play':
                played.update(card.rank for card in event[2])
            if on_event is not None:
                on_event(event)

    return engine.ranking()


def best_cards(hand, count):
    """The `count` strongest cards — what a tribute must consist of.
    The Wonder is ignored when determining "best" (it stays home)."""
    givable = [card for card in hand if card.rank != Rank.WONDER]
    givable.sort(key=lambda card: effective_strength(card.rank),
                 reverse=True)
    return givable[:count]


def weakest_cards(hand, count):
    """Default return choice for the Tycoon and the Rich: shed the
    weakest cards (never the Wonder or a Joker — they rank highest)."""
    return sorted(hand, key=lambda card: effective_strength(card.rank))[:count]


def _take(hand, cards):
    """Remove exactly these card objects (identity first — Card.__eq__
    is rank-only, so equality would grab a same-rank stand-in)."""
    for card in cards:
        for i, held in enumerate(hand):
            if held is card:
                del hand[i]
                break
        else:
            for i, held in enumerate(hand):
                if held.rank == card.rank and held.suit == card.suit:
                    del hand[i]
                    break
            else:
                raise ValueError(f"{card} is not in the hand")


def exchange_cards(hands, ranking, policies=None, on_event=None):
    """
    The between-round tribute, mutating `hands` in place.

    The Beggar's two best cards go to the Tycoon, the Poor's best to
    the Rich; each receiver returns as many freely chosen cards.

    Args:
        hands: the freshly dealt hands, indexed by seat
        ranking: last round's finish order (Tycoon first)
        policies: optional; a receiver whose policy object has an
            `exchange(hand, count)` method picks its own return cards
            (from `hand`, which already contains the tribute) instead
            of the weakest-cards default
        on_event: optional callable receiving one event per tribute:
            ('exchange', giver, receiver, tribute, returned)
    """
    tycoon, rich, poor, beggar = ranking
    for giver, receiver, count in ((beggar, tycoon, 2), (poor, rich, 1)):
        tribute = best_cards(hands[giver], count)
        _take(hands[giver], tribute)
        hands[receiver].extend(tribute)

        chooser = (getattr(policies[receiver], 'exchange', None)
                   if policies is not None else None)
        returned = (chooser(hands[receiver], count) if chooser is not None
                    else weakest_cards(hands[receiver], count))
        returned = list(returned)
        if len(returned) != count:
            raise ValueError(f"The exchange must return {count} cards, "
                             f"got {len(returned)}")
        _take(hands[receiver], returned)
        hands[giver].extend(returned)

        if on_event is not None:
            on_event(('exchange', giver, receiver, tuple(tribute),
                      tuple(returned)))


def play_game(policies, rounds, rng, on_event=None):
    """
    Run a multi-round game: deal, exchange, play, repeat.

    The 3 of Diamonds picks round one's leader; every later round
    starts with the card exchange and the Beggar leading.

    Args:
        policies: one policy per seat, used for every round
        rounds: how many rounds to play
        rng: random.Random dealing every round
        on_event: optional callable receiving, per round, a
            ('round_start', round_index, leader) event, the exchange
            events, and every engine event

    Returns:
        One ranking per round, each Tycoon-first.
    """
    rankings = []
    for index in range(rounds):
        hands = deal(rng)
        leader = (first_leader(hands) if not rankings
                  else rankings[-1][-1])
        if on_event is not None:
            on_event(('round_start', index, leader))
        roles = None
        if rankings:
            exchange_cards(hands, rankings[-1], policies, on_event)
            roles = [0] * PLAYER_COUNT
            for place, seat in enumerate(rankings[-1]):
                roles[seat] = place
            roles = tuple(roles)
        rankings.append(play_round(hands, leader, policies, on_event,
                                   roles=roles, round_index=index))
    return rankings
