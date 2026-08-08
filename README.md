# TycoonBot

This is a hobby project of mine.

I want to learn about deep learning and really like the "Tycoon" game from Persona. 

The goal of this project is, that
- **First** it will recognize the current cards in your hand as well as in play
- **Second** it will analyze which cards are still in play and who has how many cards
- **Third** it will recommend me the best possible plays to win the game
- **Fourth** it will be able to play the game alone (hopefully maybe)


## Requirements
Install dependencies:
```sh
pip install -r requirements.txt
```

Requires:
- Python 3.x
- OpenCV
- PyTorch
- NumPy


## Usage

**Live capture** — reads your hand from the game screen in a loop:
```sh
python VideoStream.py
```
Which monitor and which screen regions are used is set in `config.json`
(`monitor`: mss index, 1 = primary; regions are fractions of the frame).

**Offline test** — runs recognition on the screenshots in `Image/` and
saves annotated copies:
```sh
python ImageRecognition.py test --threshold=0.8
```

**Tests**:
```sh
pytest              # everything
pytest -m "not slow"  # skip the slow image-matching integration tests
```

## Status

Step 1 (recognizing the cards) works: template matching finds the rank
and suit glyphs in the hand (the fan needs a rotation sweep) and on the
play field, where the game conveniently dims earlier plays so only the
current trick stays readable. `GameLogic/HandReader.py` pairs the
glyphs into `Card` objects. Matching scales are cached per template
after the first confident hit, so warm frames take ~1-2s instead of ~7s.

Step 2 (tracking the game) is running: `GameLogic/GameState.py` tracks
the unseen cards per rank (the opponents' hands),
`GameLogic/PlayTracker.py` diffs consecutive play-field readings to
observe plays (own plays are recognized by cards leaving the hand and
don't touch the unseen counts), and `CardsLeftReader.py` reads each
player's "Cards Left" bubble. The top status bar — public information
the bot tracks itself anyway — serves only as ground truth:
`GameState.verify_against()` raises an alarm in the capture loop when
the bookkeeping diverges from the game.

Known gaps: the bubble digit templates only cover 0-4 and 8 (the other
digits don't appear in the reference screenshots yet), and cards
clipped at the fan edges are not read. Steps 3-4 are not started. The
`CardCNN` model in `ImageRecognition.py` is an untrained skeleton.

## License

MIT License
