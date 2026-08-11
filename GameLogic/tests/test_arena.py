import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from GameLogic.Arena import compare_games, scoreboard
from GameLogic.Simulator import random_policy


class TestCompareGames:
    def test_places_cover_every_round_of_every_game(self):
        factories = [random_policy] * 4
        places = compare_games(factories, games=2, rounds=3, seed=4)
        # 2 games x 3 rounds x 4 seats placed each round...
        assert sum(sum(counter.values()) for counter in places) == 24
        # ...and each place handed out exactly once per round.
        for place in range(4):
            assert sum(counter[place] for counter in places) == 6

    def test_games_are_reproducible(self):
        factories = [random_policy] * 4
        first = compare_games(factories, games=2, rounds=2, seed=9)
        second = compare_games(factories, games=2, rounds=2, seed=9)
        assert first == second

    def test_scoreboard_renders_game_places(self):
        factories = [random_policy] * 4
        places = compare_games(factories, games=1, rounds=2, seed=1)
        table = scoreboard(places, ['a', 'b', 'c', 'd'])
        assert 'Tycoon' in table and 'a' in table
