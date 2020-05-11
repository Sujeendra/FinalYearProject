#!/usr/bin/env python3
#from donkeycar.parts.Client import *
from sensor_msgs.msg import Image,CameraInfo
from cv_bridge import CvBridge
import sys
from donkeycar.parts.ObjectDetectionZMQ import ObjectDetection 
ros_path = sys.path.append('/usr/lib/python3/dist-packages')
if ros_path in sys.path:

    sys.path.append(ros_path)

import cv2
import rospy
import numpy as np
import imagezmq
import socket
import math

class Ecar():
  def __init__(self):
     #self.tcp=TCPClientValue('camera','192.168.1.5')
     self.imageHub = imagezmq.ImageHub()
     self.objectdetection=ObjectDetection()
     self.pub_info = rospy.Publisher('raspicam/camera_info', CameraInfo, queue_size=5)
     self.pub = rospy.Publisher('raspicam/image_raw', Image,queue_size=5)
     self.flag=0
     self.label='OK'
     self.bridge = CvBridge()
     rospy.init_node("clienTest", anonymous=True)
     self.camera_info = CameraInfo()
        # store info without header
     #self.camera_info.header = "Header"
     self.camera_info.width = int(160)
     self.camera_info.height = int(120)
     self.camera_info.distortion_model = 'plumb_bob'
     self.cx = self.camera_info.width / 2.0
     self.cy = self.camera_info.height / 2.0
     self.fx = self.camera_info.width / (
            2.0 * math.tan(float(3) * math.pi / 360.0))
     self.fy = self.fx
     self.camera_info.K = [self.fx, 0, self.cx, 0, self.fy, self.cy, 0, 0, 1]
     self.camera_info.D = [0, 0, 0, 0, 0]
     self.camera_info.R = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
     self.camera_info.P = [self.fx, 0, self.cx, 0, 0, self.fy, self.cy, 0, 0, 0, 1.0, 0]

  def start(self):
    #image,socket=self.tcp.run()
    (rpiName, image) = self.imageHub.recv_image()
    self.imageHub.send_reply(self.label.encode())

     
    if image is not None:
      imS = cv2.resize(image, (640, 480)) 
      cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL) 
      img_from_object,self.label=self.objectdetection.main(image)
      cv2.imshow('Object Detection', img_from_object)
      self.pub.publish(self.bridge.cv2_to_imgmsg(imS, "bgr8"))
      self.pub_info.publish(self.camera_info)
    if cv2.waitKey(1) & 0xff==ord('q'):
      self.flag=1
      
    #if socket is not None:
      #socket.send(b'Label')


if __name__ == '__main__':

        e_car = Ecar()
	

        while not rospy.is_shutdown():
             e_car.start()
             if e_car.flag==1:
                cv2.destroyAllWindows()
                break
#     


