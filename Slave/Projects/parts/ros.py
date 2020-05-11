import os
import rospy
from std_msgs.msg import String, Int32, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
'''


'''

class RosPublisher(object):
    '''
    A ROS node to pubish to a data stream
    '''
    def __init__(self, node_name, channel_name, stream_type=Image, anonymous=True):
        #self.data = None
        os.environ['ROS_IP'] = '192.168.1.4'

        os.environ['ROS_MASTER_URI'] = 'http://192.168.1.4:11311/'
        #os.environ['ROS_PYTHON_LOG_CONFIG_FILE'] = '|' 
        self.pub = rospy.Publisher(channel_name, stream_type,queue_size=10)
        self.bridge = CvBridge()
        rospy.init_node(node_name, anonymous=anonymous)

    def run(self, data):
        '''
        only publish when data stream changes.
        '''
        if data is not None and not rospy.is_shutdown():
            #self.data = data
            self.pub.publish(self.bridge.cv2_to_imgmsg(data, "bgr8"))
    

class RosSubscriber(object):
    '''
    A ROS node to subscribe to a data stream
    '''

    def __init__(self, node_name, channel_name, stream_type=String, anonymous=True):
        self.data = ""
        rospy.init_node(node_name, anonymous=anonymous)
        self.pub = rospy.Subscriber(channel_name, stream_type, self.on_data_recv)        

    def on_data_recv(self, data):
        self.data = data.data

    def run(self):
        return self.data

