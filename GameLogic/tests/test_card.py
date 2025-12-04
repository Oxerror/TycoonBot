import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from GameLogic.Card import Card, Rank, Suit

class TestCardCreation:   
    def test_create_card_with_suit(self):
        card = Card(Rank.ACE, Suit.SPADES)
        assert card.rank == Rank.ACE
        assert card.suit == Suit.SPADES
    
    def test_create_joker_without_suit(self):
        card = Card(Rank.JOKER)
        assert card.rank == Rank.JOKER
        assert card.suit is None
    
    def test_create_wonder_without_suit(self):
        card = Card(Rank.WONDER)
        assert card.rank == Rank.WONDER
        assert card.suit is None


class TestCardComparison:
    def test_higher_rank_wins(self):
        ace = Card(Rank.ACE, Suit.SPADES)
        king = Card(Rank.KING, Suit.HEARTS)
        assert ace > king
    
    def test_two_beats_ace(self):
        two = Card(Rank.TWO, Suit.SPADES)
        ace = Card(Rank.ACE, Suit.SPADES)
        assert two > ace
    
    def test_joker_beats_two(self):
        joker = Card(Rank.JOKER)
        two = Card(Rank.TWO, Suit.SPADES)
        assert joker > two
    
    def test_wonder_beats_joker(self):
        wonder = Card(Rank.WONDER)
        joker = Card(Rank.JOKER)
        assert wonder > joker
    
    def test_same_rank_not_greater(self):
        card1 = Card(Rank.FIVE, Suit.SPADES)
        card2 = Card(Rank.FIVE, Suit.HEARTS)
        assert not (card1 > card2)
        assert not (card2 > card1)

    def test_same_rank_is_equals(self):
        card1 = Card(Rank.FIVE, Suit.SPADES)
        card2 = Card(Rank.FIVE, Suit.HEARTS)
        assert (card1 == card2)

    def test_different_rank_is_not_equals(self):
        card1 = Card(Rank.FIVE, Suit.SPADES)
        card2 = Card(Rank.SIX, Suit.HEARTS)
        assert not (card1 == card2)
    
    def test_three_is_lowest(self):
        three = Card(Rank.THREE, Suit.SPADES)
        for rank in Rank:
            if rank != Rank.THREE:
                other = Card(rank, Suit.HEARTS) if rank not in [Rank.JOKER, Rank.WONDER] else Card(rank)
                assert other > three


class TestCardHash:
    """Test card hashing for sets and dicts"""
    
    def test_same_card_same_hash(self):
        card1 = Card(Rank.ACE, Suit.SPADES)
        card2 = Card(Rank.ACE, Suit.SPADES)
        assert hash(card1) == hash(card2)
    
    def test_different_suit_different_hash(self):
        card1 = Card(Rank.ACE, Suit.SPADES)
        card2 = Card(Rank.ACE, Suit.HEARTS)
        assert hash(card1) != hash(card2)
    
    def test_card_in_set(self):
        card = Card(Rank.ACE, Suit.SPADES)
        card_set = {card}
        assert Card(Rank.ACE, Suit.SPADES) in card_set
    
    def test_card_as_dict_key(self):
        card = Card(Rank.ACE, Suit.SPADES)
        card_dict = {card: "test"}
        assert card_dict[Card(Rank.ACE, Suit.SPADES)] == "test"
    
    def test_set_deduplication(self):
        cards = {
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS),
        }
        assert len(cards) == 2


class TestCardRepr:
    """Test card string representation"""
    
    def test_normal_card_repr(self):
        card = Card(Rank.ACE, Suit.SPADES)
        assert repr(card) == "ACE_SPADES"
    
    def test_joker_repr(self):
        card = Card(Rank.JOKER)
        assert repr(card) == "JOKER"
    
    def test_wonder_repr(self):
        card = Card(Rank.WONDER)
        assert repr(card) == "WONDER"
    
    def test_all_suits_repr(self):
        for suit in Suit:
            card = Card(Rank.KING, suit)
            assert suit.name in repr(card)


class TestCardToIndex:
    """Test card to tensor index conversion"""
    
    def test_three_of_spades_is_zero(self):
        card = Card(Rank.THREE, Suit.SPADES)
        assert card.to_index() == 0
    
    def test_two_of_spades_index(self):
        card = Card(Rank.TWO, Suit.SPADES)
        assert card.to_index() == 12  # Last card in spades (0-12)
    
    def test_three_of_hearts_index(self):
        card = Card(Rank.THREE, Suit.HEARTS)
        assert card.to_index() == 13  # First card in hearts
    
    def test_joker_index(self):
        card = Card(Rank.JOKER)
        assert card.to_index() == 53
    
    def test_wonder_index(self):
        card = Card(Rank.WONDER)
        assert card.to_index() == 54
    
    def test_all_indices_unique(self):
        """Ensure all 54 cards have unique indices"""
        indices = set()
        
        # Regular cards (52)
        for suit in Suit:
            for rank in Rank:
                if rank not in [Rank.JOKER, Rank.WONDER]:
                    card = Card(rank, suit)
                    idx = card.to_index()
                    assert idx not in indices, f"Duplicate index {idx} for {card}"
                    indices.add(idx)
        
        # Special cards
        indices.add(Card(Rank.JOKER).to_index())
        indices.add(Card(Rank.WONDER).to_index())
        
        assert len(indices) == 54
    
    def test_indices_in_valid_range(self):
        """All indices should be 0-54"""
        for suit in Suit:
            for rank in Rank:
                if rank not in [Rank.JOKER, Rank.WONDER]:
                    card = Card(rank, suit)
                    assert 0 <= card.to_index() <= 54


class TestRankOrder:
    """Test that rank values are correctly ordered"""
    
    def test_rank_order(self):
        expected_order = [
            Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN,
            Rank.EIGHT, Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN,
            Rank.KING, Rank.ACE, Rank.TWO, Rank.JOKER, Rank.WONDER
        ]
        for i in range(len(expected_order) - 1):
            assert expected_order[i].value < expected_order[i + 1].value


class TestSuit:
    """Test suit enum"""
    
    def test_four_suits(self):
        assert len(Suit) == 4
    
    def test_suit_names(self):
        suit_names = [s.name for s in Suit]
        assert "SPADES" in suit_names
        assert "HEARTS" in suit_names
        assert "DIAMONDS" in suit_names
        assert "CLUBS" in suit_names