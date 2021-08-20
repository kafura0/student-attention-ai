REAL-TIME FINE-GRAINED HEAD POSE ESTIAMTION USING DEEP LEARNING
The original work: [GitHub](http://github.com/natanielruiz/deep-head-pose)
This repo was a part of my Nottingham University Thesis.

![Run Demo](final_demo.gif)

This repository shows how to perform a head pose estimation in real time video footage. 

The project was intended to solve the problem of drowsy driving and thus a major requirement was that it works in "in the wild" situations. This determined the choise of the libraries used.

The following publication discusses in detail the research of the topic and the considerations involved: https://www.researchgate.net/publication/334286261_Convolution_Neural_Networks_for_Head_pose_estimation_in_the_wild

 ## LIBRARIES USED
* Dockerface 
* deep head pose 
* PyTorch
* FLask

##REQUIREMENTS
* numpy
* pytorch
* torchvision
* opencv2
* pillow

## Flow of the code 
* The image is passed to head pose estimator on the server , the server gets face bounding box predictions from dockerface running on localhost:5000 on server , the predictions are used to get head pose pitch ,yaw and roll.
* These values are used to issue warning if out of limit.
* The detected face bounding box and frame is passd to eye_closed_utils.py , which detects face features and finds the distance between the eye lids.
* Threshold is set to 0.25 if value less then that then eyes are closed.



           

