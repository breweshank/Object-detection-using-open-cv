import cv2
import numpy as np

# Define the color ranges in HSV for red, green, and blue
color_ranges = {
    'Red': [np.array([0, 120, 70]), np.array([10, 255, 255]), np.array([170, 120, 70]), np.array([180, 255, 255])],
    'Green': [np.array([40, 52, 72]), np.array([80, 255, 255])],
    'Blue': [np.array([94, 80, 2]), np.array([126, 255, 255])]
}

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    counts = {'Red': 0, 'Green': 0, 'Blue': 0}

    for color, ranges in color_ranges.items():
        if color == 'Red':
            mask1 = cv2.inRange(hsv, ranges[0], ranges[1])
            mask2 = cv2.inRange(hsv, ranges[2], ranges[3])
            mask = mask1 | mask2
        else:
            mask = cv2.inRange(hsv, ranges[0], ranges[1])

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Get the area of the contour
            area = cv2.contourArea(contour)

            # Ignore small areas
            if area > 500:
                counts[color] += 1
                # Get the bounding box coordinates of the contour
                x, y, w, h = cv2.boundingRect(contour)

                # Draw the bounding box around the detected object
                if color == 'Red':
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                elif color == 'Green':
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                elif color == 'Blue':
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f'{color} Object', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the counts on the frame
    cv2.putText(frame, f"Red: {counts['Red']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Green: {counts['Green']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Blue: {counts['Blue']}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
