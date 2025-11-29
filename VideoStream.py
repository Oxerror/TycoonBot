import cv2
import numpy as np
import mss

def videoCapturing():
	sct = mss.mss()

	# monitors[0] is a virtual monitor representing all monitors
	# monitors[1] is the full primary monitor
	# monitors[2], [3], ... are individual monitors
	monitor = sct.monitors[2]  # Full primary monitor
      
	while True:
		img = np.array(sct.grab(monitor))

		frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
		cv2.imshow('Screen Capture', frame)

		# Exit on 'q' key press
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	# Clean up
	cv2.destroyAllWindows()

if __name__ == "__main__":
	videoCapturing()
