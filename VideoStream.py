import time

import cv2
import mss

from ImageRecognition import HAND_MATCH_PARAMS, get_recognizer, read_play_field
from GameLogic.Card import Card
from GameLogic.GameState import DECK_SIZE, GameState, validate_start_hand
from GameLogic.HandReader import detections_to_cards, hand_is_ordered
from GameLogic.PlayTracker import PlayTracker
from GameLogic.Recommender import recommend
from CardsLeftReader import read_cards_left_detailed
from ScreenCapture import applyRedactions, cropRegion, getScreen, loadConfig
from StatusBarReader import read_status_bar


def drawDetections(image, detections):
    """Draw bounding boxes and labels for detected cards."""
    output = image.copy()

    # Colors for different card types (BGR format)
    colors = {
        # Numbers
        '2': (0, 255, 0), '3': (0, 255, 0), '4': (0, 255, 0), '5': (0, 255, 0),
        '6': (0, 255, 0), '7': (0, 255, 0), '8': (0, 255, 0), '9': (0, 255, 0), '10': (0, 255, 0),
        # Face cards
        'Jack': (255, 0, 255), 'Queen': (255, 0, 255), 'King': (255, 0, 255), 'Ace': (255, 0, 255),
        # Suits
        'Heart': (0, 0, 255), 'Diamond': (0, 0, 255),
        'Spade': (255, 255, 0), 'Cross': (255, 255, 0),
        # Special
        'Joker': (0, 255, 255), 'Wonder': (0, 255, 255),
    }
    default_color = (128, 128, 128)

    for detection in detections:
        name = detection['name']
        confidence = detection['confidence']
        x, y, w, h = detection['location']

        color = colors.get(name, default_color)

        # Draw bounding box
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        # Draw label background
        label = f"{name} ({confidence:.0%})"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(output, (x, y - label_h - 6), (x + label_w + 4, y), color, -1)

        # Draw label text
        cv2.putText(output, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return output


def videoCapturing():
    config = loadConfig()
    recognizer = get_recognizer()
    sct = mss.mss()

    # Tracked state, initialized from the first readable status bar and
    # kept current by observed plays. The bar is only read as ground
    # truth: a persistent divergence means the bot messed up.
    game_state = None
    tracker = None
    diverged_frames = 0

    print("Capturing... press 'q' in a window to quit.")

    while True:
        frame = getScreen(sct, config['monitor'])
        applyRedactions(frame, config['redact_regions'])

        playField = cropRegion(frame, config['play_field'])
        currentHand = cropRegion(frame, config['hand_region'])

        start = time.time()
        detections = recognizer.template_match(currentHand, **HAND_MATCH_PARAMS)
        cards = detections_to_cards(detections)
        trick = read_play_field(playField)
        elapsed = time.time() - start

        handWithDetections = drawDetections(currentHand, detections)

        print(f"Hand ({elapsed:.1f}s): {cards}")
        if not hand_is_ordered(cards):
            # The game always displays the hand sorted, so an unordered
            # reading proves at least one card was misrecognized.
            print("WARNING: hand reading is out of display order - likely a misread")
        if trick:
            print(f"Current trick: {trick}")

        bar_counts = read_status_bar(frame)
        counters, _, active_player = read_cards_left_detailed(frame)

        if bar_counts is None:
            print("Status bar: not visible")
        elif game_state is None:
            game_state = GameState.from_status_bar(bar_counts)
            tracker = PlayTracker(game_state)
            tracker.update(trick, cards, counters['player'])
            print(f"Tracking started: {game_state.total_unseen()} unseen cards")
        else:
            try:
                for event in tracker.update(trick, cards, counters['player']):
                    who = 'we' if event['by_player'] else 'opponent'
                    print(f"Play observed ({who}): {event['cards']}")
            except ValueError as error:
                print(f"ALARM: impossible play observed - {error}")

            mismatches = game_state.verify_against(bar_counts)
            if not mismatches:
                diverged_frames = 0
                print("State verified: tracking matches the game")
            else:
                # One divergent frame can be a play caught mid-animation;
                # a persistent one means the bot lost track of the game.
                diverged_frames += 1
                diff = ', '.join(f"{r.name} {t}->{a}"
                                 for r, (t, a) in mismatches.items())
                if diverged_frames >= 2:
                    print(f"ALARM: bot state diverged from the game ({diff})")
                    print("Re-syncing from the status bar.")
                    game_state = GameState.from_status_bar(bar_counts)
                    tracker = PlayTracker(game_state)
                    tracker.update(trick, cards, counters['player'])
                    diverged_frames = 0
                else:
                    print(f"State mismatch this frame ({diff}), waiting one frame")

            opponent_counts = [counters[k] for k in ('left', 'middle', 'right')]
            if None not in opponent_counts:
                opponents_total = sum(opponent_counts)
                if opponents_total != game_state.total_unseen():
                    print(f"Cards-left cross-check: bubbles say {opponents_total}, "
                          f"tracking says {game_state.total_unseen()}")

            # At round start nothing has been played, so deck - bar must
            # equal the own hand — validates the hand reading, reveals
            # the clipped fan-edge cards and gives the tracker complete
            # knowledge of the own hand.
            all_counts = list(counters.values())
            if None not in all_counts and sum(all_counts) == DECK_SIZE:
                missing, extra = validate_start_hand(cards, bar_counts)
                if extra:
                    print("WARNING: hand reading shows cards the bar rules out: "
                          + ", ".join(f"{r.name} x{n}" for r, n in extra.items()))
                else:
                    recovered = [Card(rank) for rank, n in missing.items()
                                 for _ in range(n)]
                    tracker.set_known_hand(cards + recovered)
                    print(f"Round start: full hand known: {tracker.known_hand_cards()}"
                          + (f" ({len(recovered)} recovered from the bar)"
                             if recovered else ""))

            # The active player's bubble carries a red marker; suggest a
            # move only when it is ours.
            if tracker.revolution:
                print("REVOLUTION is active - strength order is flipped")
            if active_player == 'player':
                own_hand = tracker.known_hand_cards() if tracker.known_hand else cards
                if own_hand:
                    move = recommend(own_hand, trick, tracker.revolution)
                    print(f"YOUR TURN - suggested play: {list(move) if move else 'PASS'}")
            elif active_player is not None:
                print(f"Waiting: {active_player} opponent is playing")

        cv2.imshow('Field', playField)
        cv2.imshow('Hand', handWithDetections)

        # imshow windows stay responsive only while waitKey pumps events
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    videoCapturing()
