# IMPROVED-VEHICLE-COUNTING-WITH-OCCLUSION-RESOLUTION-ON-MONOCLUAR-TRAFFIC-VIDEO-SEQUENCES
This project is a computer vision based vehicle counting system. The system is capable of performing vehicle detetcion, tracking and counting of vehicles using feature based detection. The project is also capable of handling occlusion in the case of two vehicles. The project aims at providing better results while counting of vehicles in a given scenario.

## Getting Started
Download a copy of the project onto the system using a web browser or terminal commands. Unzip the porject and you are good to go.

### Prerequisites
Python v3.5+ <br />
OpenCV - Open Source Computer Vision v3.4+  <br />
Anaconda (Create a separate environment for your project) <br />

Use the following commands to install the packages into your environment: <br />
conda env create -f environment.yaml <br />
source activate cv <br />

Or you can install everything globally. Search for step by step guides to install OpenCV. The dependencies will be installed on the way. <br />

## Files in the box
Why do I see so many files and what are their roles? <br />
Here's an overview of the files. <br />

Feature based detection: <br />
DetectionUsingFAST.py : Vehicle detetcion using FAST features <br />
DetectionUsingSIFT.py : Vehicle detetcion using SIFT features <br />
DetectionUsingSURF.py : Vehicle detetcion using SURF features <br />
DetectionUsingORB.py : Vehicle detetcion using ORB features <br />
Feature tracking: <br />
FeatureTracking.py : Tracking all the features detected in the frame<br />
FeatureTrackingCenterOfCluster.py : Tracking the centre of the clusters detected in the frame by clustering the features using hierarichal clustering <br />
FeatureTrackingFullVehicle.py : Tracking the cluster of the vehicle <br />
OcclusionHandling.py : Handling the occlusion problem by separating two vehicles using a dividing line<br /> 
All the files have the module for feature tracking and occlusion handling. 

## Let's run this thing

Activate your environment. Change the directory to the project Folder. Create an Input and Results folder. Place the input video in the Input folder. Run the file using python <br />
Example: <br />
python OcclusionHandling.py <br />
python FeatureTrackingFullVehicle.py <br />
