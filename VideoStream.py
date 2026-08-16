import argparse
import time

import cv2
import mss
import numpy as np

from InputExecutor import InputExecutor
from ScreenCapture import applyRedactions, getScreen, loadConfig
from Session import TycoonSession


def buildExecutor(act):
    """Suggest-only unless a pad kind was explicitly requested.

    This is the single place an act-mode executor is constructed; it
    exists behind the --act flag so pressing real buttons is always a
    deliberate choice, never a default.
    """
    if act is None:
        return InputExecutor()
    from VirtualGamepad import DS4Backend, XboxBackend
    backend = DS4Backend() if act == 'ds4' else XboxBackend()
    print(f"ACT MODE: planned buttons WILL be pressed on a virtual {act} pad.")
    return InputExecutor(backend=backend, mode='act')


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


def parkDebugWindows(sct, capture_monitor):
    """Create the debug windows on a monitor that is not captured.

    A window sitting over the game feeds the loop its own (stale)
    output: every reading it covers goes back in time, which corrupts
    tracking. With a single monitor the windows stay where the OS puts
    them and must be dragged clear of the captured regions by hand.
    """
    others = [monitor for index, monitor in enumerate(sct.monitors)
              if index > 0 and index != capture_monitor]
    for name, x_offset, y_offset in (('Bot', 20, 20), ('Field', 740, 20),
                                     ('Hand', 20, 560)):
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        if others:
            cv2.moveWindow(name, others[0]['left'] + x_offset,
                           others[0]['top'] + y_offset)
    if not others:
        print("WARNING: single monitor - drag the windows off the game "
              "regions or they corrupt every reading they cover")


def renderMessages(messages, suggestion):
    """Status panel image: the current suggestion big, the frame log below."""
    canvas = np.zeros((500, 680, 3), np.uint8)
    cv2.putText(canvas, 'Suggested play:', (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(canvas, suggestion if suggestion else '(waiting)', (12, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    y = 120
    for message in messages:
        cv2.putText(canvas, message[:78], (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
        y += 22
    return canvas


def videoCapturing(executor=None, policy='search'):
    config = loadConfig()
    session = TycoonSession(config, executor=executor, policy=policy)
    sct = mss.mss()
    parkDebugWindows(sct, config['monitor'])

    print("Capturing... press 'q' in a window to quit.")

    suggestion = None
    while True:
        frame = getScreen(sct, config['monitor'])
        applyRedactions(frame, config['redact_regions'])

        start = time.time()
        messages, detections, (hand_region, field_region) = session.process_frame(frame)
        elapsed = time.time() - start

        print(f"--- frame ({elapsed:.1f}s)")
        for message in messages:
            print(message)
            if 'YOUR TURN' in message:
                suggestion = message.split(': ', 1)[1]
            elif 'Waiting:' in message or 'Play observed (we)' in message:
                suggestion = None

        cv2.imshow('Bot', renderMessages(messages, suggestion))
        cv2.imshow('Field', field_region)
        # Halved so the full-width hand strip fits beside the game monitor.
        cv2.imshow('Hand', cv2.resize(drawDetections(hand_region, detections),
                                      None, fx=0.5, fy=0.5,
                                      interpolation=cv2.INTER_AREA))

        # imshow windows stay responsive only while waitKey pumps events
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Follow the game live and suggest moves.")
    parser.add_argument(
        '--act', choices=('xbox', 'ds4'),
        help="press the planned buttons on a virtual pad (ds4 for PS "
             "Remote Play) instead of only logging them")
    parser.add_argument(
        '--policy', choices=('search', 'net'), default='search',
        help="who decides on trusted turns: the rollout search "
             "(default) or the trained net in one forward pass")
    args = parser.parse_args()
    videoCapturing(buildExecutor(args.act), policy=args.policy)
