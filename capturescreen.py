import cv2
import numpy
import time
import mss
import torch
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from pynput import keyboard
from PIL import Image
from ultralytics import YOLO


###################### CONFIG ######################

# Declare yolov11 model
model = YOLO("yolo11n.pt")

# Area we dont want to detect objects in (These objects are still detected just flagged)
exemption_zone = Polygon([(0, 800), (340, 340), (470, 340), (800, 800)])

# Area of the screen to capture
monitor = {"top": 240, "left": 560, "width": 800, "height": 600}

###################### CONFIG ######################


# Check for CUDA/GPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")  # Use CPU
    print("Using CPU")


def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf, device=device)

    else:
        results = chosen_model.predict(img, conf=conf)

    return results


def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:

            #Check if detected object is in exclusion zone. 
            middle = ((int(box.xyxy[0][0]) + int(box.xyxy[0][2])) / 2, (int(box.xyxy[0][1]) + int(box.xyxy[0][3])) / 2)
            middle = Point(middle)

            if exemption_zone.contains(middle):
                color = (255, 0, 0)
            
            else:
                color = (0, 128, 0)

            cv2.rectangle(img, 
                         (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                         (int(box.xyxy[0][2]), int(box.xyxy[0][3])), 
                         color, rectangle_thickness)

            cv2.putText(img, 
                        f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, color, text_thickness)

    return img, results


def on_press(key):
    global active_key

    try:
        active_key = key.char

    except AttributeError:
        active_key = key


def on_release(key):
    global active_key

    active_key = "No Action"

'''
    if key == keyboard.Key.esc:
        # Stop listener
        return False
'''


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

time.sleep(1)
with mss.mss() as sct:
    active_key = none

    # Capture loop
    while "Screen capturing":
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))

        # Convert to RGB color space
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect objects in our frame
        img, _ = predict_and_detect(model, img, classes=[9, 2, 7, 11, 0], conf=0.2)

        # Get FPS
        fps = "FPS: " + str(round(1 / (time.time() - last_time)))

        # Draw FPS over image
        img = cv2.putText(img, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        img = cv2.putText(img, str(active_key), (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw exclusion zone
        e_zone = numpy.array([[0, 800],[300, 310],[500, 310],[800, 800]], numpy.int32)
        e_zone = e_zone.reshape((-1,1,2))
        img = cv2.polylines(img, [e_zone], True, (255, 0, 0), 2)

        # Display the picture
        cv2.imshow("Big Smoke Uber", img)

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break