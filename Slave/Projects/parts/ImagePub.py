import imagezmq
import socket
import time
"""
Used to publish images to Master for object Recognition 
"""
#from imutils.video import VideoStream
class ImagePublish:
  def __init__(self):
    self.sender = imagezmq.ImageSender(connect_to="tcp://192.168.1.7:5555")
    self.rpiName = socket.gethostname()
    self.imageHub = imagezmq.ImageHub()
    #self.vs = VideoStream(usePiCamera=True).start()

  def run(self,image):
    #frame=self.vs.read()
    #self.sender = imagezmq.ImageSender(connect_to="tcp://192.168.1.7:5555")
    if(image is not None and self.sender is not None):
        print("Connection successful")
        print(self.sender.send_image(self.rpiName,image))
        #print(self.imageHub.zmq_socket.recv())

        
  def shutdown(self):
    pass
    
