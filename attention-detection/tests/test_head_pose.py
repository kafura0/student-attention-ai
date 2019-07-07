import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_pose(self):
        import requests
        import json
        import cv2

        addr = 'http://35.185.247.42'
        test_url = addr + '/api/face_identify'

        # prepare headers for http request
        content_type = 'image/jpeg'
        headers = {'content-type': content_type}

        img = cv2.imread('4.jpeg')
        # encode image as jpeg
        _, img_encoded = cv2.imencode('.jpg', img)
        # send http request with image and receive response
        response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
        # decode response
        print(json.loads(response.text))

    def test_video(self):
        import cv2
        import requests
        import json
        cap = cv2.VideoCapture(0)

        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            addr = 'http://35.185.247.42'
            test_url = addr + '/api/face_identify'

            # prepare headers for http request
            content_type = 'image/jpeg'
            headers = {'content-type': content_type}

            # img = cv2.imread('test.png')
            # encode image as jpeg
            _, img_encoded = cv2.imencode('.jpg', frame)
            # send http request with image and receive response
            response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
            # decode response
            print(json.loads(response.text))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    def test_eye_closed(self):
        from eye_closed_utils import get_eye_closed_warning
        import cv2
        img = cv2.imread('test.png')
        print(get_eye_closed_warning(img))


if __name__ == '__main__':
    unittest.main()
