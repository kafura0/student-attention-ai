import json

import cv2
import hopenet
import numpy as np
import requests
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
from PIL import Image
from flask import Flask, request, jsonify
from torch.autograd import Variable
from torchvision import transforms

cudnn.enabled = True

batch_size = 1
gpu = 0

# load hopnet model
model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

print("Load pre-trained model")
# Load snapshot
saved_state_dict = torch.load("face_pose_attention_detection_80k_iter.pkl")
nt=model.load_state_dict(saved_state_dict)
print(nt)
print('Loading data.')

transformations = transforms.Compose([transforms.Scale(224),
                                      transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

model.cuda(gpu)

print('Ready to test network.')

# Test the Model
model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
total = 0

idx_tensor = [idx for idx in range(66)]
idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

addr = 'http://localhost:5000'
test_url = addr + '/api/test'
content_type = 'image/jpeg'
headers = {'content-type': content_type}
app = Flask(__name__)


@app.route('/api/face_identify', methods=['POST'])
def test():
    r = request
    nparr = np.fromstring(r.data, np.uint8)
    imgs = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame = imgs
    cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Get face co-ordinates from dockerface
    try:
        response = requests.post(test_url, data=r.data, headers=headers)
    except Exception as ex:
        print("Exception in dockerface endpoint please check")
        return jsonify({"Error": "exception in dockerface endpoint {0}".format(str(ex))})
    data = json.loads(response.text)
    print("Predictions from dockerface of face co-odinates ")
    for i in data:
        print(i)
    # Pick max confidence face
    maxconfidence_face = max(data, key=lambda x: x['confidence_score'])
    print("max confidence face co-odinates")
    print(maxconfidence_face)

    x_min, y_min, x_max, y_max = int(float(maxconfidence_face["x_min"])), int(float(maxconfidence_face["y_min"])), int(
        float(maxconfidence_face["x_max"])), int(float(maxconfidence_face["y_max"]))

    bbox_width = abs(x_max - x_min)
    bbox_height = abs(y_max - y_min)
    # x_min -= 3 * bbox_width / 4
    # x_max += 3 * bbox_width / 4
    # y_min -= 3 * bbox_height / 4
    # y_max += bbox_height / 4
    x_min -= 50
    x_max += 50
    y_min -= 50
    y_max += 30
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(frame.shape[1], x_max)
    y_max = min(frame.shape[0], y_max)
    # Crop face loosely
    img = cv2_frame[y_min:y_max, x_min:x_max]
    img = Image.fromarray(img)

    # Transform
    img = transformations(img)
    img_shape = img.size()
    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
    img = Variable(img).cuda(gpu)

    yaw, pitch, roll = model(img)

    yaw_predicted = F.softmax(yaw)
    pitch_predicted = F.softmax(pitch)
    roll_predicted = F.softmax(roll)
    # Get continuous predictions in degrees.
    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99
    yaw_predicted=yaw_predicted.item()
    pitch_predicted=pitch_predicted.item()
    roll_predicted=roll_predicted.item()
    return jsonify(
        {"yaw_predicted": yaw_predicted, "pitch_predicted": pitch_predicted, "roll_predicted": roll_predicted,
         "max_confidence_Detected_face": maxconfidence_face})
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6000)
