import cv2
from app.vision.camera import Camera
from app.vision.hand_detector import HandDetector
from app.vision.landmark_extractor import LandmarkExtractor
from app.vision.preprocessor import Preprocessor
from app.utils.image import draw_fps, draw_finger_states, draw_bounding_box
import time

preprocessor = Preprocessor(flip_horizontal=True)
extractor = LandmarkExtractor()
prev_time = time.time()

with Camera() as cam, HandDetector() as detector:
    for frame in cam.stream():
        frame = preprocessor.process(frame)
        result = detector.detect(frame)
        annotated = detector.draw(frame, result)

        for hand in result.hands:
            features = extractor.extract(hand)
            draw_bounding_box(annotated, features.bounding_box, label=features.hand_label)
            draw_finger_states(annotated, features.fingers)

        fps = 1.0 / (time.time() - prev_time + 1e-9)
        prev_time = time.time()
        draw_fps(annotated, fps)

        cv2.imshow("LESCO-AI", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()