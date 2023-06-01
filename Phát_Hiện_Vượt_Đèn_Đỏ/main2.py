import cv2
import numpy as np

# Define range of red color in HSV
lower_red = np.array([0,100,100])
upper_red = np.array([10,255,255])
lower_red2 = np.array([160,100,100])
upper_red2 = np.array([180,255,255])
# Open video file
cap = cv2.VideoCapture('test1.mp4')
# Set up background subtractor

fgbg = cv2.createBackgroundSubtractorMOG2()

# Set up kernel for morphologqical operations
kernel = np.ones((5, 5), np.uint8)

# Set up variables for tracking cars
cars = []
car_id = 0
counter = 0
while cap.isOpened():
    # Capture frame-by-frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    ret, frame = cap.read()

    if not ret:
        break

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    size = frame.shape
    # Create mask for red color
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.add(mask1, mask2)
    # Apply contour detection
    r_circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 80,
                                 param1=50, param2=10, minRadius=0, maxRadius=30)

    r = 5
    bound = 4.0 / 10
    # Draw the line
    if r_circles is not None:
        r_circles = np.uint16(np.around(r_circles))
        fgmask = fgbg.apply(frame)
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        line_pos = 220
        # line_pos=300
        # cv2.line(frame, (0, line_pos), (size[1], line_pos), (0, 0, 255), thickness=2)
        for cnt in contours:
            # Calculate area of contour
            area = cv2.contourArea(cnt)

            # Filter out contours that are too small
            if area < 1000:
                continue

            # Get bounding box of contour
            x, y, width, height = cv2.boundingRect(cnt)

            # Calculate aspect ratio of bounding box
            aspect_ratio = float(width) / height

            # Filter out bounding boxes that are too long or too short
            if aspect_ratio > 2.5 or aspect_ratio < 0.4:
                continue
            # cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        for i in r_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1] + m) >= size[0] or (i[0] + n) >= size[1]:
                        continue
                    h += mask[i[1] + m, i[0] + n]
                    s += 1
            if h / s > 50:
                cv2.circle(frame, (i[0], i[1]), i[2] + 10, (0,0, 255), 2)
                cv2.circle(mask, (i[0], i[1]), i[2] + 30, (255, 255, 255), 2)
                cv2.putText(frame, 'RED', (i[0], i[1]), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if y + height > line_pos and y < line_pos:
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame, 'Xe vuot den do', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    # Display the resulting frame
    frame_resized = cv2.resize(frame, (1280, 720))
    cv2.imshow('frame',  frame_resized)

    # Exit when "q" is pressed
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
