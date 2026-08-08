import time

import cv2
import mss

from ScreenCapture import applyRedactions, getScreen, loadConfig
from Session import TycoonSession


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
    session = TycoonSession(config)
    sct = mss.mss()

    print("Capturing... press 'q' in a window to quit.")

    while True:
        frame = getScreen(sct, config['monitor'])
        applyRedactions(frame, config['redact_regions'])

        start = time.time()
        messages, detections, (hand_region, field_region) = session.process_frame(frame)
        elapsed = time.time() - start

        print(f"--- frame ({elapsed:.1f}s)")
        for message in messages:
            print(message)

        cv2.imshow('Field', field_region)
        cv2.imshow('Hand', drawDetections(hand_region, detections))

        # imshow windows stay responsive only while waitKey pumps events
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    videoCapturing()
