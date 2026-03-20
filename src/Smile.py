import mediapipe as mp
import cv2 as cv
import time # Needed for timestamps
import numpy as np
import math

# Setup mediapipe aliases
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None  # Global variable to store the latest result

# 1. Define the Callback function FIRST
# It MUST take these 3 arguments: result, output_image, timestamp_ms
def result_callback(result, output_image, timestamp_ms):
	global latest_result
	latest_result = result

# 2. Setup face landmark model with the callback included
options = FaceLandmarkerOptions(
	base_options=BaseOptions(model_asset_path="face_landmarker.task"),
	running_mode=VisionRunningMode.LIVE_STREAM,
	num_faces=1,
	result_callback=result_callback # <--- This was the missing piece!
)

def calculate_smile_length(mouth_left, mouth_right):
	return math.sqrt((mouth_right.x - mouth_left.x) ** 2 + (mouth_right.y - mouth_left.y) ** 2)

start_time = time.time()
calibration_duration = 7  # seconds
calibration_data = []
original_smile_length = None
# 3. Initialize the model
with FaceLandmarker.create_from_options(options) as landmarker:
	cap = cv.VideoCapture(0)
	
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
		
		# 4. Generate a timestamp in milliseconds
		timestamp = int(time.time() * 1000)
		landmarker.detect_async(mp_image, timestamp) #estudar essa linha dps

		if latest_result  and latest_result.face_landmarks:
			face = latest_result.face_landmarks[0]  # Get the first detected face
			mouth_left = face[61]
			mouth_right = face[291]
			mouth_width = calculate_smile_length(mouth_left, mouth_right)

			cheek_left = face[234]
			cheek_right = face[454]

			static_width = calculate_smile_length(cheek_left, cheek_right)
			curent_ratio = mouth_width/static_width

		
# --- TIMER LOGIC ---
			elapsed_time = time.time() - start_time

			if elapsed_time < 7:
				# PHASE 1: CALIBRATING
				calibration_data.append(curent_ratio)
				cv.putText(frame, f"CALIBRATING: {int(7 - elapsed_time)}s", (50, 50), 
						cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
				
			else:	
				# PHASE 2: MONITORING
				if original_smile_length is None:
					# Calculate the baseline average once
					original_smile_length = sum(calibration_data) / len(calibration_data) if calibration_data else 0.0001
					print(f"Calibration complete: {original_smile_length:.4f}")

				# --- THIS PART MUST BE OUTSIDE THE 'IF NONE' BLOCK ---
				# Calculate the ratio (1.0 = no change, 1.2 = 20% wider)
				smile_factor = curent_ratio / original_smile_length     

				# Logic: If mouth is 10% wider than the neutral baseline
				if smile_factor > 1.10: 
					status = "Smiling"
					color = (0, 255, 0) # Green
				else:
					status = "Not Smiling"
					color = (0, 0, 255) # Red

				# DRAW the status on the screen
				cv.putText(frame, f"Status: {status}", (10, 30), 
						cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
				cv.putText(frame, f"Factor: {smile_factor:.2f}x", (10, 70), 
						cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
			for landmark in face:
				x_pixel = int(landmark.x*frame.shape[1])
				y_pixel = int(landmark.y*frame.shape[0])
				cv.circle(frame,(x_pixel, y_pixel), 1, (35,20,30), -1)

	

			mouth_left = face[61]
			mouth_right = face[291]
			mx = int(mouth_left.x * frame.shape[1])
			my = int(mouth_left.y * frame.shape[0])
			rx = int(mouth_right.x * frame.shape[1])
			ry = int(mouth_right.y * frame.shape[0])

			#def line(img: cv2.typing.MatLike, pt1: cv2.typing.Point, pt2: cv2.typing.Point, color: cv2.typing.Scalar, thickness: int = ..., lineType: int = ..., shift: int = ...) -> cv2.typing.MatLike: ...

			#smile_line = cv.line(frame, (mx, my), (rx, ry), (50, 30, 200), 2)
			
			smile_len = np.linalg.norm(np.array([mx, my]) - np.array([rx, ry]))

			cv.imshow("Face Tracking", frame)
			
		if cv.waitKey(1) & 0xFF == ord('q'):
			break


	cap.release()
	cv.destroyAllWindows()
 
"""
	The above Python code uses the MediaPipe library to detect facial landmarks, calibrate a smile
	length, and monitor the user's smile status in real-time using webcam input.
	
	:param result: The `result` parameter in the code snippet refers to the result obtained from the
	face landmark detection process. It contains information about the detected face landmarks such as
	the positions of key points on the face like eyes, nose, and mouth. This information is used to
	calculate various metrics like smile length and facial
	:param output_image: The `output_image` parameter in the callback function refers to the image frame
	that is processed by the face landmark model. In the provided code snippet, the `output_image` is
	not explicitly used within the callback function. However, you can utilize this parameter to access
	the processed image if needed for any
	:param timestamp_ms: The `timestamp_ms` parameter in the code represents the timestamp in
	milliseconds. It is used to keep track of the time when certain events occur, such as when
	processing frames from a video stream or detecting landmarks on a face. This timestamp is essential
	for synchronization and time-based operations within the application
	"""
 
 #author: Lis Peixoto Almeida