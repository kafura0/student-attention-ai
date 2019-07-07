# attention-detection-pose-based
Driver attention detection using deep learning based on head pose .


Setting up 
- Run dockerface NV_GPU=0 nvidia-docker run -p 5000:5000 dockerface_flask:latest
- Run deep head pose flask server python3 run_flask.py, in deep head pose directory 
- Run main.py by specifying path of image file
#### Flow of the code is as follows
- The image is passed to head pose estimator on the server , the server gets face bounding box predictions from dockerface running on localhost:5000 
on server , the predictions are used to get head pose pitch ,yaw and roll.
- These values are used to issue warning if out of limit.
- The detected face bounding box and frame is passd to eye_closed_utils.py , which detects face features and finds the distance between the eye lids.
- Threshold is set to 0.25 if value less then that then eyes are closed.
- Values can be calibrated further if need be.


#### Docker face setup
- cd into docker_face , build docker image. Expose port 5000 and run , now flask server for face detection running on port 5000.
-docker build command : docker build -t dockerface_flask .
- then NV_GPU=0 nvidia-docker run -p 5000:5000 dockerface_flask:latest

#### deep head pose
- CUDA should be installed for both docker face and deep head pose on the server .
- Download the model file in the docker_face directory from the following link.
```
https://storage.googleapis.com/dataspecc-models/face_pose_attention_detection_80k_iter.pkl
```
- python3 run_flask.py , now server running on port 6000.
- Use local code to make requests .

##### Using sample data 
- change path of image to run for that specific one in main.py.
- Take a slefie with eyes closed towards the camera , warning will be issued in terminal .
- Take a selfie with face either face tilting down or above warning will be issued based on pitch of the face .
- Check on other data values of prediciton are printed in console .
- Can be tinkered with , right now limit is at +25 degree and -25 degree.