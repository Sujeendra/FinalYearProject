#!/usr/bin/env python3
from donkeycar.parts.path import Path,PImage,PathPlot,PlotCircle,PImage,CTE
from geometry_msgs.msg import PoseStamped
from donkeycar.parts.transform import PIDController
import sys
import numpy as np
from PIL import Image
ros_path = sys.path.append('/usr/lib/python3/dist-packages')
if ros_path in sys.path:

    sys.path.append(ros_path)
import rospy
import cv2
class PathPlanning:
  def __init__(self):
    rospy.init_node("pathComputation", anonymous=True)
    self.path_ob=Path()
    self.load=self.path_ob.load("Pathforodometry")
    self.PathPlot_ob=PathPlot()
    self.circle=PlotCircle()
    self.img_ob=PImage()
    self.pid=PIDController(p=2,i=1,d=1)
    self.x=0
    self.y=0
    #self.img=cv2.imread("/home/sujeendra/index.png")
    self.im=None
    #self.im = Image.fromarray((self.img * 255).astype(np.uint8))
    self.im_1=self.im
    self.im_2=self.im
    self.cte=CTE()
    rospy.Subscriber('/mono_odometer/pose', PoseStamped, self.odometry_callback)
  
  def odometry_callback(self,msg):
     #print("Callback")
     self.x=msg.pose.position.x
     self.y=msg.pose.position.y
    
  def main(self):
    #print(self.x,self.y)
    self.path_ob.run(self.x,self.y)
    self.im=self.img_ob.run()
    self.im_1=self.im
    self.im_2=self.im
    
    if self.load is not None:
      print(self.pid.run(self.cte.run(self.load,self.x,self.y)))
    if self.path_ob.path is not None and self.im is not None:
      #print('Image not None')
      #self.path_ob.save("Pathforodometry")
      self.im_1=self.PathPlot_ob.run(self.im_1,self.path_ob.path)
      self.im_2=self.circle.run(self.im_2,self.x,self.y)
      cv2.imwrite('PathComputation.png',np.array(self.im_1))
      cv2.imwrite('PlotCircle.png',np.array(self.im_2))
     
if __name__ == '__main__':
  ob=PathPlanning()
  while not rospy.is_shutdown(): 
     ob.main()
#ROS_NAMESPACE=raspicam rosrun image_proc image_proc
#roslaunch gscam mycam.launch
#rosrun finalyearproject clienTest.py 
#rosrun viso2_ros mono_odometer image:=/raspicam/image_rect
# rostopic echo /mono_odometer/pose
#rosrun finalyearproject pathComputation.py
#
