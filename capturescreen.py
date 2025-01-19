import cv2
import numpy
import time
import mss
import keyboard
from ultralytics import YOLO

# Declare yolov11 model
model = YOLO("yolo11n.pt")

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)

    else:
        results = chosen_model.predict(img, conf=conf)

    return results


def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, 
                         (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                         (int(box.xyxy[0][2]), int(box.xyxy[0][3])), 
                         (255, 0, 0), rectangle_thickness)

            cv2.putText(img, 
                        f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)

    return img, results



with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {"top": 240, "left": 560, "width": 800, "height": 600}

    # Capture loop
    while "Screen capturing":
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img, _ = predict_and_detect(model, img, classes=[9, 2, 7, 11, 0], conf=0.5)

        # Get FPS
        fps = "FPS: " + str(round(1 / (time.time() - last_time)))

        #Draw FPS over image
        img = cv2.putText(img, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the picture
        cv2.imshow("Big Smoke Uber", img)

        # Display the picture in grayscale
        # cv2.imshow('OpenCV/Numpy grayscale',
        #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

        #print(f"fps: {1 / (time.time() - last_time)}")

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break