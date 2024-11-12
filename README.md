# Simple Motion Detection with OpenCV
Description:
This Python script utilizes OpenCV to implement basic motion detection in a video stream. It captures video from the default camera (webcam), detects significant changes between frames, and highlights them as potential motion events.
Features:
Leverages OpenCV for computer vision tasks.
Detects motion by calculating the absolute difference between frames and applying thresholding.
Filters out small objects to minimize noise.
Displays a bounding box around detected objects.
Prints the current state (Normal or Moving Object Detected) to the console.
Allows quitting the program by pressing the 'y' key.
Installation:
1.Python installed (python3 --version).
2.Install OpenCV using pip (pip install opencv-python).
3.Install imutils (optional, for frame resizing): pip install imutils
