"""
Deep head pose estimation
"""
import json

import cv2
import dlib
import requests

from eye_closed_utils import get_eye_closed_warning


def make_prediction(image_path):
    addr = 'http://35.185.247.42'
    test_url = addr + '/api/face_identify'

    # prepare headers for http request
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    img = cv2.imread(image_path)
    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', img)
    # send http request with image and receive response
    response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
    # decode response
    return json.loads(response.text), img


def get_pose(path):
    """
    :param img: path to the image to make prediction for
    :return: json payload containing the warning and its parameters
    """
    # static limits for warning they can be calibrated accordingly
    # The relevant axis of head movement is pitch (upward movement ) and roll (tilting the head)
    # The limit is 30 degree you can change while testing
    LIMIT_POSITIVE = 25
    LIMIT_NEGATIVE = -25
    try:
        print("Payload sent to server for prediction")
        predictions, img = make_prediction(path)
        print("Detected face stats {0}".format(predictions["max_confidence_Detected_face"]))
        pitch = predictions["pitch_predicted"]
        roll = predictions["roll_predicted"]
        print("pitch is {0} and roll is {1}".format(pitch, roll))
        if pitch > LIMIT_POSITIVE or pitch < LIMIT_NEGATIVE:
            print("Warning raised due to pitch being overlimit pitch :{0}".format(pitch))
        if roll > LIMIT_POSITIVE or roll < LIMIT_NEGATIVE:
            print("Warning raised due to roll being overlimit roll :{0}".format(roll))
        x = float(predictions["max_confidence_Detected_face"]['x_min'])
        y = float(predictions["max_confidence_Detected_face"]['y_min'])
        w = float(predictions["max_confidence_Detected_face"]['x_max'])
        h = float(predictions["max_confidence_Detected_face"]['y_max'])
        print(x, y, w, h)
        print("bbox tuple for face feature localization {0}".format((x, y, w, h)))
        bb_box = (x, y, w, h)

        dt = dlib.rectangle(int(x), int(y), int(w), int(h))
        # check if eyes closed
        try:
            print("Payload sent to eye_closed_utils for check eyes are closed")
            warning, frame, eye_ratio = get_eye_closed_warning(img, dt)
            # Display frame for visualization if needed
            if warning:
                print("Warning driver is drowsy because eyes close {0} as aspect ratio is {1} and limit is 0.25".format(
                    warning, eye_ratio))
        except Exception as ex:
            print("Face shape not detected {0}".format(ex))


    except Exception as ex:
        print("Exception in processing image is : {0}".format(ex))


if __name__ == '__main__':
    # pass the image here you wna tto get predicitons for
    get_pose("2.jpeg")
