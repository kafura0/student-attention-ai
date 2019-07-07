from __future__ import division
import _init_paths

import os

import caffe
import cv2
import numpy as np
from fast_rcnn.config import cfg
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.test import im_detect
from flask import Flask, request, jsonify

# Dockerface network
NETS = {'vgg16': ('VGG16',
                  'output/faster_rcnn_end2end/wider/vgg16_dockerface_iter_80000.caffemodel')}

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    # cfg.TEST.BBOX_REG = False

    prototxt = os.path.join(cfg.MODELS_DIR, NETS['vgg16'][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS['vgg16'][1])

    prototxt = 'models/face/VGG16/faster_rcnn_end2end/test.prototxt'
    caffemodel = NETS['vgg16'][1]

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    caffe.set_mode_gpu()
    caffe.set_device(0)
    cfg.GPU_ID = 0
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print
    '\n\nLoaded network {:s}'.format(caffemodel)

    # Initialize the Flask application
    app = Flask(__name__)


    # Coding for API endpoint
    @app.route('/api/test', methods=['POST'])
    def test():
        r = request
        CONF_THRESH = 0.7
        NMS_THRESH = 0.15
        nparr = np.fromstring(r.data, np.uint8)

        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # img is BGR cv2 image.
        # # Detect all object classes and regress object bounds
        scores, boxes = im_detect(net, img)

        cls_ind = 1
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        keep = np.where(dets[:, 4] > CONF_THRESH)
        dets = dets[keep]

        # dets are the upper left and lower right coordinates of bbox
        # dets[:, 0] = x_ul, dets[:, 1] = y_ul
        # dets[:, 2] = x_lr, dets[:, 3] = y_lr

        dets[:, 2] = dets[:, 2]
        dets[:, 3] = dets[:, 3]
        if (dets.shape[0] != 0):
            output = []
            for j in xrange(dets.shape[0]):
                # Write file_name bbox_coords
                # fid.write(args.image_path.split('/')[-1] + ' %f %f %f %f %f\n' % (dets[j, 0], dets[j, 1], dets[j, 2], dets[j, 3], dets[j, 4]))
                output.append({"frame_number": j, "x_min": dets[j, 0], "y_min": dets[j, 1], "x_max": dets[j, 2],
                               "y_max": dets[j, 3], "confidence_score": dets[j, 4]})
        return jsonify(output)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
