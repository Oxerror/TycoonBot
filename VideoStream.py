import cv2
import numpy as np
import mss

def getScreen():
	sct = mss.mss()

	# monitors[0] is a virtual monitor representing all monitors
	# monitors[1] is the full primary monitor
	# monitors[2], [3], ... are individual monitors
	monitor = sct.monitors[1]
	  
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

def videoCapturing():
	frame = getScreen()

	height, width = frame.shape[:2]
	playField = frame[height*2//5:height*4//5, width*1//3:width*2//3]
	currentHand = frame[height*3//4:height]

	cv2.imshow('Field', playField)  
	cv2.imshow('Hand', currentHand)

	# used for debugging purposes
	while True:
	# Exit on 'q' key press
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	# Clean up
	cv2.destroyAllWindows()

if __name__ == "__main__":
	videoCapturing()
