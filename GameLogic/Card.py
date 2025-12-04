from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

class Suit(Enum):
    SPADES = auto()
    HEARTS = auto()
    DIAMONDS = auto()
    CLUBS = auto()


class Rank(Enum):
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    SEVEN = 5
    EIGHT = 6
    NINE = 7
    TEN = 8
    JACK = 9
    QUEEN = 10
    KING = 11
    ACE = 12
    TWO = 13
    JOKER = 14
    WONDER = 15


@dataclass
class Card:
    rank: Rank
    suit: Optional[Suit] = None  # Joker / Wonder has no suit
    
    def __hash__(self):
        return hash((self.rank, self.suit))
  
    def __gt__(self, other):
        return self.rank.value > other.rank.value
    
    def __eq__(self, other):
        return self.rank.value == other.rank.value
    
    def __repr__(self):
        if self.suit:
            return f"{self.rank.name}_{self.suit.name}"
        return f"{self.rank.name}"
    
    def to_index(self) -> int:
        if self.rank == Rank.JOKER:
            return 53
        if self.rank == Rank.WONDER:
            return 54

        suit_offset = list(Suit).index(self.suit) * 13
        rank_offset = self.rank.value - 1
        return suit_offset + rank_offset