# "IN THE WILD" DRIVER ATTENTION DETECTION USING DEEP LEARNING

This repository shows how to perferm a head pose estimation in real time video footage. 

The project was intended to solve the problem of drowsy driving and thus a major requirement was that it works in "in the wild" situations. This determined the choise of the libraries used.

The following publicsation discusses in detail the research of the topic and the considerations involved: https://www.researchgate.net/publication/334286261_Convolution_Neural_Networks_for_Head_pose_estimation_in_the_wild

 ## LIBRARIES USED
* Dockerface - Face Detection Docker Solution Using Faster R-CNN
* deep head pose - ResNet50
* PyTorch
* FLask

## Flow of the code 
* The image is passed to head pose estimator on the server , the server gets face bounding box predictions from dockerface running on localhost:5000 on server , the predictions are used to get head pose pitch ,yaw and roll.
* These values are used to issue warning if out of limit.
* The detected face bounding box and frame is passd to eye_closed_utils.py , which detects face features and finds the distance between the eye lids.
* Threshold is set to 0.25 if value less then that then eyes are closed.



           

