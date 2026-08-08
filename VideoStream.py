import cv2
import numpy as np
import mss
from ImageRecognition import get_recognizer

def getScreen():
	sct = mss.mss()

	# monitors[0] is a virtual monitor representing all monitors
	# monitors[1] is the full primary monitor
	# monitors[2], [3], ... are individual monitors
	monitor = sct.monitors[2]
	  
	img = np.array(sct.grab(monitor))
	frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

	# Find all white areas in the image
	# Define lower and upper bounds for white color
	lower_white = np.array([230, 230, 230], dtype=np.uint8)
	upper_white = np.array([255, 255, 255], dtype=np.uint8)

	# Create a mask for white regions
	white_mask = cv2.inRange(frame, lower_white, upper_white)

	highlighted = np.zeros_like(frame)  # Start with a black image
	highlighted[white_mask == 255] = [255, 255, 255]  # Set whiteish pixels to white

	return highlighted

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
	recognizer = get_recognizer()
	
	frame = getScreen()

	height, width = frame.shape[:2]
	playField = frame[height*2//5:height*4//5, width*1//3:width*2//3]
	currentHand = frame[height*3//4:height]

	# Recognize cards from the current hand (mask already applied in getScreen)
	detections = recognizer.template_match(currentHand, threshold=0.8, apply_mask=False)
	
	# Draw detections on the hand image
	handWithDetections = drawDetections(currentHand, detections)
	
	# Extract detected names for console output
	numbers = [d['name'] for d in detections]
	print(f"Detected cards: {numbers}")

	cv2.imshow('Field', playField)  
	cv2.imshow('Hand', handWithDetections)

	# used for debugging purposes
	while True:
	# Exit on 'q' key press
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	# Clean up
	cv2.destroyAllWindows()

if __name__ == "__main__":
	videoCapturing()
