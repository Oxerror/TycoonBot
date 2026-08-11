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

**Replay a recorded session** — validates the whole pipeline against
the frames in `Image/captures/`:
```sh
python Replay.py
```
Recognition results are cached in `*.readings.json` sidecars next to
the frames, so only the first replay pays for template matching —
after that a full session replays in seconds. Delete the sidecars to
force fresh recognition.

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

Step 3 is running: `GameLogic/Rules.py` encodes the Tycoon rules (equal
rank sets with jokers as wildcards, revolution, 8-stop, the 3-Spade
Reversal against a single Joker, and the Wonder winning any trick) and
`GameLogic/Recommender.py` is the heuristic baseline — finish when
possible, lead weak, win cheap, save power cards — made unseen-aware:
`Rules.is_unbeatable()` proves from the tracked counts when no opponent
can top a set, and the recommender plays out mathematically won rounds
(a chain of unbeatable sets) on the spot. The active player's bubble
carries a red marker, which `CardsLeftReader` reads as turn detection,
so the loop suggests a move exactly when it is your turn. Revolution
state comes from the persistent "Flip Strength" badge the game shows
above the player box, read every frame.

On top of that, `GameLogic/TrickEngine.py` runs whole rounds (turn
order, pass lock-outs, trick resolution, finish order),
`GameLogic/Simulator.py` deals seeded hands and pits pluggable
policies against each other, and `GameLogic/SearchRecommender.py`
picks the live suggestion by determinized rollout search: sample
deals of the unseen cards consistent with the opponents' Cards Left
counters, play every candidate move to the end of the round, keep the
one with the best average finish. A player who passed keeps yellow
bubble text until the trick ends; `CardsLeftReader` reads it, so the
search knows who is locked out and who owns the current set.

The simulator also chains rounds into whole games
(`Simulator.play_game`): after round one the Beggar leads, and each
later round opens with the card exchange — the Beggar's two best
cards to the Tycoon, the Poor's best to the Rich, freely chosen cards
back (weakest by default; a policy can override the choice).
`Arena.compare_games` measures policies across such games, where
winning a round buys better cards in the next.

The first learned player exists too: `python -m GameLogic.SelfPlay`
has four search policies play seeded rounds against each other,
records every sampled decision (situation, candidate move, average
finish over the rollouts), and distills them into a small evaluator
net (`GameLogic/PolicyNet.py`, saved as `policy_net.pt`). In the
arena (`python -m GameLogic.Arena`): the heuristic wins 82% against
three random players; the search wins 38% against three heuristics;
the learned net — one forward pass, no rollouts — wins 44% against
three heuristics, though the search still outranks it at a shared
table (37% vs 22%). Next rung of the ladder: put the net back into
the search as its rollout policy and iterate.

Known gaps: in dense fans the outermost cards are clipped beyond their
emblems and are not read (recovered at round start via the bar), and
identical overlapping cards (a double-joker play) count as one. The
`CardCNN` model in `ImageRecognition.py` is an untrained skeleton.

## License

MIT License
