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

By default the loop only *suggests*: it prints the recommended move and
the button presses that would play it. Add `--act xbox` or `--act ds4`
(PS Remote Play only accepts PlayStation pads) to actually press them
on a virtual gamepad — each turn is pressed at most once, and the pad
requires the ViGEmBus driver (`pip install vgamepad` offers it).

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

**Virtual gamepad smoke test** — proves the input path with the game
NOT running:
```sh
python VirtualGamepad.py          # emulate an Xbox 360 pad
python VirtualGamepad.py --ds4    # emulate a DualShock 4 (PS Remote Play)
```
Requires `pip install vgamepad` (its setup offers the ViGEmBus driver).
The script presses every mapped button once per second; watch the
emulated pad respond in Windows' controller panel (Win+R → `joy.cpl` →
select the controller → Properties) or at
https://hardwaretester.com/gamepad. The pad only joins the game loop
when `VideoStream.py` is started with `--act`; without the flag
`InputExecutor` stays suggest-only.

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

The hand read survives the game's own-turn UI: cards that cannot join
a valid play are dimmed to a uniform grey, so a second matching pass
inside that dim band (`read_hand_detections`) recovers them, and the
hand crop reaches above the resting fan so cards lifted by selection
stay in view. On the field, a confident rank glyph whose suit symbol a
neighbor card covers still counts (suitless), keeping the trick size —
how many cards a play must contain — honest.

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
(a chain of unbeatable sets) on the spot. Turn detection reads the
Pass/Hint button row the game draws while waiting for the player (the
red bubble marker, its fallback for the row's fade-in, washes out on
two-digit counts), so the loop suggests a move exactly when it is your
turn. Revolution
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
the learned net — one forward pass, no rollouts — wins 37% against
three heuristics.

The ladder's first rung is climbed (`--ladder`): the trained net goes
back into the search as its rollout policy — fed each rollout seat's
full view of the determinized world, which matters: steering blind
distilled into a clearly weaker net — and the strengthened search
(average place 1.36 vs 1.55 for its heuristic-rollout peers)
regenerates the training data. Retrained on all data pooled, the new
net beats its predecessor head-to-head, 29% vs 21% Tycoon over 400
shared-table rounds, while keeping the heuristic matchup. Training on
the ladder data alone lost that matchup (19%): nets calibrate to the
play style behind their targets, so the pool keeps both styles.

Step 4 (playing alone) has its offline half: `InputPlanner.py`
translates a suggested move into the button presses that would play it
— D-pad steps across the recognized fan (bar-recovered edge cards get
slotted in by the game's display sort), select each card, confirm, or
hit pass — with every unverified UI belief (cursor start/wrap, button
roles, confirm flow) isolated in its `UI_ASSUMPTIONS` block for the
first live session to correct in one place. `Session` runs the planner
on every own turn and hands the sequence to an `InputExecutor`, which
in its default suggest-only mode just logs it; Replay therefore
exercises the whole loop — read → track → search → plan inputs — on
the recorded captures, and the planned sequences are fixtured beside
them like the recognition readings
(`tests/test_input_plan_replay.py`). The act half is
`VirtualGamepad.py` (Xbox 360 via vgamepad/ViGEm, DualShock 4 for PS
Remote Play), smoke-testable without the game and reachable only via
`VideoStream.py --act`, which presses each turn's plan at most once;
verifying the UI assumptions and calibrating press timings against the
real game remains a supervised live-session task.

Known gaps: in dense fans the outermost cards are clipped beyond their
emblems and are not read (recovered at round start via the bar),
identical overlapping cards (a double-joker play) count as one, and a
lifted (selected) card whose glyph the neighboring lifted card covers
stays unread until deselected — the tracker carries it meanwhile, since
the Cards Left counter proves it never left the hand. The `CardCNN`
model in `ImageRecognition.py` is an untrained skeleton.

## License

MIT License
