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

Step 1 (recognizing the cards in your hand) works: template matching
finds the rank and suit glyphs (the fanned hand needs a rotation sweep),
and `GameLogic/HandReader.py` pairs them into `Card` objects.

`StatusBarReader.py` reads the remaining-cards-per-rank bar at the top
of the screen. This is public information the bot will later track
itself from observed plays, so the reader is only a verification tool
to check the tracked state against the game.

Steps 2-4 are not started. The `CardCNN` model in
`ImageRecognition.py` is an untrained skeleton for later.

## License

MIT License
