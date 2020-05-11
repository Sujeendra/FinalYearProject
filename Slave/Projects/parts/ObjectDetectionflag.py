import time
import os
import cv2
import numpy as np
import sys
sys.path.append('/home/pi/mycar')
import test as m
class ObjectDetectionflag(object):
    def __init__(self):
        self.flag=True
        m.init()
        self.throttle_coeff = 1.0       
        self.max_dist = 50.0            
        self.slow_down_dist = 13.0      # when car starts slowing down
        self.stop_dist = 11.0           # when car has to stop
        self.stop_time = 5            # stop time in seconds
        self.steering_coeff=1.0
        self.have_stopped = False       # car has responded to the current stop
        self.to_sleep = False           # need to sleep this tim
        self.classifier = os.path.join("/home/pi/projects/parts/cv/stopsign_classifier.xml")
        
    
    def stop_sign_detection(self, image_array):
        '''
        return 0 if no stop sign was detected, or
        return area of largest stop sign detected.
        '''
        area = 0
        if image_array is not None:
            classifier = cv2.CascadeClassifier(self.classifier)
            image_array_np = np.array(image_array)
            gray = cv2.cvtColor(image_array_np, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('gray',gray)
            stop_signs = classifier.detectMultiScale(image=gray, scaleFactor=1.02, minNeighbors=10)
            #print(len(stop_signs), "STOPPED")
            for (x, y, w, h) in stop_signs:
                area = max(w * h, area)
                print("-- Area: ", area)
                #print("x: ", x, " | y: ", y, " | w: ", w, " | h: ", h)
        if area!=0:
           self.flag=False
           m.count=1
        if m.x==False:
           m.x=True
           self.flag=True
                
        return self.flag

        
    def run(self,image_array):
        return self.stop_sign_detection(image_array)

    
    def shutdown(self):
        pass
