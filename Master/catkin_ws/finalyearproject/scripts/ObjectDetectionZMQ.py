#!/usr/bin/env python3
import sys
ros_path = sys.path.append('/usr/lib/python3/dist-packages')
if ros_path in sys.path:
   sys.path.append(ros_path)
import rospy
from sensor_msgs.msg import Image
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2
import zmq
import time
from cv_bridge import CvBridge



class ObjectDetection:
     def __init__(self):
        rospy.init_node("ObjectDetection", anonymous=True)
        

        self.ob_image=None
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-p", "--prototxt", required=True,help="path to Caffe 'deploy' prototxt file")
        self.ap.add_argument("-m", "--model", required=True,help="path to Caffe pre-trained model")
        self.ap.add_argument("-c", "--confidence", type=float, default=0.2,help="minimum probability to filter weak detections")
        self.ap.add_argument("-mW", "--montageW", required=True, type=int,help="montage frame width")
        self.ap.add_argument("-mH", "--montageH", required=True, type=int,help="montage frame height")
        self.args = vars(self.ap.parse_args())   
        #rospy.Subscriber('/raspicam/image_raw', Image, self.Image_callback)  
        #self.pub = rospy.Publisher('objectdetection/view', Image,queue_size=5)
        self.bridge = CvBridge()
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"] 
        # load our serialized model from disk
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(self.args["prototxt"], self.args["model"])
        self.CONSIDER = set(["dog", "person", "car"])
        #self.objCount = {obj: 0 for obj in CONSIDER}
        self.mW = self.args["montageW"]
        self.mH = self.args["montageH"]
        print("[INFO] detecting: {}...".format(", ".join(obj for obj in
	self.CONSIDER)))

     '''def Image_callback(self,image_message):
         cv_image = self.bridge.imgmsg_to_cv2(image_message, "bgr8")
         self.ob_image=cv_image'''
         
     def main(self,get_image):
       self.ob_image=get_image
       if self.ob_image is not None:
         frame = imutils.resize(self.ob_image, width=400)
         (h, w) = frame.shape[:2]
         blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)
         self.net.setInput(blob)
         detections = self.net.forward()
         objCount = {obj: 0 for obj in self.CONSIDER}
         for i in np.arange(0, detections.shape[2]):
		
                confidence = detections[0, 0, i, 2]
                if confidence > self.args["confidence"]:
			
                        idx = int(detections[0, 0, i, 1])

			# check to see if the predicted class is in the set of
			# classes that need to be considered
                        if self.CLASSES[idx] in self.CONSIDER:
				# increment the count of the particular object
				# detected in the frame
                                objCount[self.CLASSES[idx]] += 1

				# compute the (x, y)-coordinates of the bounding box
				# for the object
                                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                (startX, startY, endX, endY) = box.astype("int")

				# draw the bounding box around the detected object on
				# the frame
                                cv2.rectangle(frame, (startX, startY), (endX, endY),(255, 0, 0), 2)
         cv2.putText(frame, "Object Detection Frame", (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
         
	   # draw the object count on the frame
         label = ", ".join("{}: {}".format(obj, count) for (obj, count) in objCount.items())
         print(label)
         cv2.putText(frame, label, (10, h - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)
         #cv2.imshow('frame',frame)
         print(type(self.bridge.cv2_to_imgmsg(frame, "bgr8")))
         if self.bridge.cv2_to_imgmsg(frame, "bgr8") is not None:
             cv2.imwrite('image.png',frame)
             #self.pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
       return frame


'''if __name__=='__main__':
   ob=ObjectDetection()
   while not rospy.is_shutdown():
       ob.main()'''
#rosrun finalyearproject clienTest.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel --montageW 2 --montageH 2
#rosrun image_view image_view image:=/objectdetection/view


