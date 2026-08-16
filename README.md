# OpenCV-Only Hand Gesture Recognition

Python 3.14 compatible.

Install:
py -3.14 -m pip install -r requirements.txt

Run:
py -3.14 gesture_recognition.py

Press Q to quit.

Pipeline:
Webcam -> OpenCV -> Skin Segmentation -> Hand Contour
-> Convexity Defects -> Finger Count -> Gesture -> Output

Gestures:
0 = FIST
1 = ONE
2 = TWO
3 = THREE
4 = FOUR
5 = FIVE

No MediaPipe, TensorFlow, or Keras is required.

For best results use good lighting and a plain background.
