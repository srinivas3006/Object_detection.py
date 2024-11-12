import cv2 
import time
import imutils

# Initialize video capture from the default camera (0)
cam = cv2.VideoCapture(0)

# Wait for the camera to warm up
time.sleep(1)

# Initialize the first frame (background)
firstFrame = None

# Set a minimum area threshold for detected objects
area = 500

while True:
    # Read a frame from the camera
    _, img = cam.read()

    # Set default text for normal state
    text = "Normal"

    # Resize the image for efficiency
    img = imutils.resize(img, width=500)

    # Convert the image to grayscale for easier processing
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)

    # If the first frame is not set, initialize it
    if firstFrame is None:
        firstFrame = gaussianImg
        continue

    # Calculate the difference between the current frame and the background
    imgDiff = cv2.absdiff(firstFrame, gaussianImg)

    # Apply thresholding to isolate significant differences
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill gaps in objects
    threshImg = cv2.dilate(threshImg, None, iterations=2)

    # Find contours in the thresholded image
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Iterate through each contour
    for c in cnts:
        # Filter out small contours that are likely noise
        if cv2.contourArea(c) < area:
            continue

        # Get the bounding box coordinates of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # Draw a bounding box around the detected object
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Update the text to indicate motion detection
        text = "Moving Object Detected"

    # Print the current state to the console
    print(text)

    # Add the text to the image
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the processed image
    cv2.imshow("cameraFeed", img)

    # Wait for a key press
    key = cv2.waitKey(10)
    print(key)

    # Exit the loop if the 'y' key is pressed
    if key == ord("y"):
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()