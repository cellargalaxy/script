from time import sleep

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path="yolov8n_saved_model/yolov8n_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture("rtsp://rtspstream:jG2zm1ADvRayJ7XLoiBT5@zephyr.rtsp.stream/people")

while True:
    ret, frame = cap.read()
    if not ret:
        print('break')
        break

    img = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    # input_data = np.expand_dims(img, axis=0).astype(np.uint8)
    input_data = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    has_person = any(obj[0] == 0 and obj[1] > 0.5 for obj in output_data[0])

    print("has_person",has_person)
    sleep(1)

    # cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
