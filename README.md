# Hand Gesture Recognition

A Python webcam application that detects faces and hands, estimates hand landmarks, counts extended fingers, and displays a simple gesture label in real time. OpenCV provides camera/video display and face detection; MediaPipe Hands provides 21-point hand landmarks.

## Gestures

The current mapping reports a finger count from zero through five as `FIST`, `ONE`, `TWO`, `THREE`, `FOUR`, or `FIVE`. Results depend on camera angle, lighting, hand visibility, and the landmark detector; this is a demonstration, not a reliable control or identity system.

## Requirements

Use a Python version supported by the pinned MediaPipe dependency (the source comments recommend Python 3.12). Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

Connect a webcam, then run:

```bash
python gesture_recognition.py
```

Press **Q** to close the application. By default, the script opens camera index 0; update `CAMERA_INDEX` in the script if needed.

## Privacy

Video frames are processed locally by the application and are not intentionally saved by the provided script. Obtain consent before using a camera around other people and review the licensing/terms of OpenCV and MediaPipe.

## License

No license is currently declared for this repository. Unless a license is added, all rights are reserved; obtain permission before reusing or redistributing the code.
