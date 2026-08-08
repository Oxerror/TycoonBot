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

**Collect training data** — run alongside the game to gather real
frames for tests and templates:
```sh
python CaptureData.py
```
Saves one frame per play (triggered by status-bar changes) plus frames
whose "Cards Left" counters show digits the templates don't know yet,
to `Image/captures/` (gitignored) with a JSON sidecar of auto-labels.
The `redact_regions` in `config.json` are blacked out before saving so
the platform user id never reaches the disk — calibrate the region
once against a live frame before publishing captured images.

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

All 56 cards are recognized (four per rank plus two Jokers and two
Wonders; the specials match via their distinctive emblem art). The
game always displays the hand sorted (Wonder, 3..Ace, 2, Joker), which
`hand_is_ordered()` uses as a misread detector, and at round start
deck - bar = own hand, so `validate_start_hand()` proves the hand
reading correct and recovers the cards clipped at the fan edges.

Step 3 has begun: `GameLogic/Rules.py` encodes the Tycoon rules (equal
rank sets with jokers as wildcards, revolution, 8-stop, the 3-Spade
Reversal against a single Joker, and the Wonder winning any trick) and
`GameLogic/Recommender.py` is a first heuristic recommender — finish
when possible, lead weak, win cheap, save power cards — whose
suggestion the capture loop prints each frame. Turn detection and a
stronger (search/learning based) recommender are next.

Known gaps: in dense fans the outermost cards are clipped beyond their
emblems and are not read (recovered at round start via the bar), and
identical overlapping cards (a double-joker play) count as one. The
`CardCNN` model in `ImageRecognition.py` is an untrained skeleton.

## License

MIT License
