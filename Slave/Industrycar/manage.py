#!/usr/bin/env python3
"""

"""
import sys
import os
import time
import zmq
import socket
import zlib, pickle
#import zmq
import select
sys.path.append('/home/pi/mycar')
import test as m
from datetime import datetime
from docopt import docopt
import numpy as np

import donkeycar as dk
from donkeycar.parts.object_detection.objecdetectiondonkeycar_py import ObjectDetectionModel
from donkeycar.parts.ImagePub import ImagePublish
#import parts
#from donkeycar.parts.flag import Flag
from donkeycar.parts.transform import Lambda, TriggeredCallback, DelayedTrigger
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.controller import LocalWebController, JoystickController
from donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.parts.behavior import BehaviorPart
#from donkeycar.parts.stop import Stop
from donkeycar.parts.file_watcher import FileWatcher
from donkeycar.parts.launch import AiLaunch
from donkeycar.parts.foo import Foo
from donkeycar.utils import *
from donkeycar.parts.ObjectDetectionflag import ObjectDetectionflag

def drive(cfg, model_path=None, use_joystick=False, model_type=None, camera_type='single', meta=[] ):
    '''
    Construct a working robotic vehicle from many parts.
    Each part runs as a job in the Vehicle loop, calling either
    it's run or run_threaded method depending on the constructor flag `threaded`.
    All parts are updated one after another at the framerate given in
    cfg.DRIVE_LOOP_HZ assuming each part finishes processing in a timely manner.
    Parts may have named outputs and inputs. The framework handles passing named outputs
    to parts requesting the same named input.
    '''

    
    
    #Initialize car
    V = dk.vehicle.Vehicle()

    
    if cfg.CAMERA_TYPE == "PICAM":
            from donkeycar.parts.camera import PiCamera
            cam = PiCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        
    else:
            raise(Exception("Unkown camera type: %s" % cfg.CAMERA_TYPE))
            
    V.add(cam, inputs=inputs, outputs=['cam/image_array'], threaded=threaded)
        
    
    ctr = LocalWebController()

    #V.add(Flag(), inputs=['cam/image_array'], outputs=['flag'])


   
    V.add(ctr, 
          inputs=['cam/image_array'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
          threaded=True)
    img_pub=ImagePublish()
    V.add(img_pub,inputs=['cam/image_array'])
    class TCPServeValue(object):
       def __init__(self):
          self.s = socket.socket() 
          self.port = 5555 
          self.s.connect(('192.168.1.7', self.port))
       def run(self):
          data=self.s.recv(1024)
          if data is not None:
             print("Data from Server",data)

       def shutdown(self):
          self.s.close()
    #tcp=TCPServeValue()
    #V.add(tcp)

    class ValuePrinting:
      def __init__(self):
         pass
      def run(self,angle,throttle,mode,recording,flag):
          print("angle: ",angle)
          print(" throttle: ",throttle)
          print(" mode: ",mode)
          print(" recording: ",recording)
          print("flag: ",flag)
          #print("client: ",client)
    V.add(ValuePrinting(),inputs=['user/angle', 'user/throttle', 'user/mode', 'recording','flag'])
   

    #this throttle filter will allow one tap back for esc reverse
    #th_filter = ThrottleFilter()
    #V.add(th_filter, inputs=['user/throttle'], outputs=['user/throttle'])
    
    #See if we should even run the pilot module. 
    #This is only needed because the part run_condition only accepts boolean
    class PilotCondition:
        def run(self, mode):
            if mode == 'user':
                return False
            else:
                return True       

    V.add(PilotCondition(), inputs=['user/mode'], outputs=['run_pilot'])
    
    
    class ImgPreProcess():
        '''
        preprocess camera image for inference.
        normalize and crop if needed.
        '''
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self, img_arr):
            return normalize_and_crop(img_arr, self.cfg)

    if "coral" in model_type:
        inf_input = 'cam/image_array'
    else:
        inf_input = 'cam/normalized/cropped'
        V.add(ImgPreProcess(cfg),
            inputs=['cam/image_array'],
            outputs=[inf_input],
            run_condition='run_pilot')

   

    def load_model(kl, model_path):
        start = time.time()
        print('loading model', model_path)
        kl.load(model_path)
        print('finished loading in %s sec.' % (str(time.time() - start)) )

    def load_weights(kl, weights_path):
        start = time.time()
        try:
            print('loading model weights', weights_path)
            kl.model.load_weights(weights_path)
            print('finished loading in %s sec.' % (str(time.time() - start)) )
        except Exception as e:
            print(e)
            print('ERR>> problems loading weights', weights_path)

    def load_model_json(kl, json_fnm):
        start = time.time()
        print('loading model json', json_fnm)
        from tensorflow.python import keras
        try:
            with open(json_fnm, 'r') as handle:
                contents = handle.read()
                kl.model = keras.models.model_from_json(contents)
            print('finished loading json in %s sec.' % (str(time.time() - start)) )
        except Exception as e:
            print(e)
            print("ERR>> problems loading model json", json_fnm)

    if model_path:
        #When we have a model, first create an appropriate Keras part
        kl = dk.utils.get_model_by_type(model_type, cfg)

        model_reload_cb = None

        if '.h5' in model_path:
            #when we have a .h5 extension
            #load everything from the model file
            load_model(kl, model_path)

            def reload_model(filename):
                load_model(kl, filename)

            model_reload_cb = reload_model

        
        else:
            print("ERR>> Unknown extension type on model file!!")
            return

        #this part will signal visual LED, if connected
        V.add(FileWatcher(model_path, verbose=True), outputs=['modelfile/modified'])

        #these parts will reload the model file, but only when ai is running so we don't interrupt user driving
        V.add(FileWatcher(model_path), outputs=['modelfile/dirty'], run_condition="ai_running")
        V.add(DelayedTrigger(100), inputs=['modelfile/dirty'], outputs=['modelfile/reload'], run_condition="ai_running")
        V.add(TriggeredCallback(model_path, model_reload_cb), inputs=["modelfile/reload"], run_condition="ai_running")

        outputs=['pilot/angle', 'pilot/throttle']

    
        V.add(kl, inputs=inputs, 
            outputs=outputs,
            run_condition='run_pilot')            
    
    #Choose what inputs should change the car.
    class DriveMode:
        def run(self, mode, 
                    user_angle, user_throttle,
                    pilot_angle, pilot_throttle):
            if mode == 'user': 
                return user_angle, user_throttle
            
            elif mode == 'local_angle':
                return pilot_angle, user_throttle
            
            else: 
                return pilot_angle, pilot_throttle * cfg.AI_THROTTLE_MULT
        
    V.add(DriveMode(), 
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'], 
          outputs=['angle', 'throttle'])
    
    #to give the car a boost when starting ai mode in a race.
    aiLauncher = AiLaunch(cfg.AI_LAUNCH_DURATION, cfg.AI_LAUNCH_THROTTLE, cfg.AI_LAUNCH_KEEP_ENABLED)
    
    V.add(aiLauncher,
        inputs=['user/mode', 'throttle'],
        outputs=['throttle'])

    if isinstance(ctr, JoystickController):
        ctr.set_button_down_trigger(cfg.AI_LAUNCH_ENABLE_BUTTON, aiLauncher.enable_ai_launch)
        
        
    #V.add(ObjectDetectionModel(),inputs=['cam/image_array'],outputs=['angle'])
    #V.add(ObjectDetectionModel(), inputs=['cam/image_array'],threaded=True)
    #stop sign detector
    #V.add(Stop(), inputs=['throttle','angle','cam/image_array','user/mode'], outputs=['throttle','angle','user/mode'])


    class AiRunCondition:
        '''
        A bool part to let us know when ai is running.
        '''
        def run(self, mode):
            if mode == "user":
                return False
            return True

    V.add(AiRunCondition(), inputs=['user/mode'], outputs=['ai_running'])

    #Ai Recording
    class AiRecordingCondition:
        '''
        return True when ai mode, otherwize respect user mode recording flag
        '''
        def run(self, mode, recording):
            if mode == 'user':
                return recording
            return True

    if cfg.RECORD_DURING_AI:
        V.add(AiRecordingCondition(), inputs=['user/mode', 'recording'], outputs=['recording'])
    
    #Drive train setup
    
    if cfg.DRIVE_TRAIN_TYPE == "SERVO_ESC":
        from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle

        steering_controller = PCA9685(cfg.STEERING_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
        steering = PWMSteering(controller=steering_controller,
                                        left_pulse=cfg.STEERING_LEFT_PWM, 
                                        right_pulse=cfg.STEERING_RIGHT_PWM)
        
        throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
        throttle = PWMThrottle(controller=throttle_controller,
                                        max_pulse=cfg.THROTTLE_FORWARD_PWM,
                                        zero_pulse=cfg.THROTTLE_STOPPED_PWM, 
                                        min_pulse=cfg.THROTTLE_REVERSE_PWM)

        V.add(steering, inputs=['angle'])
        V.add(throttle, inputs=['throttle'])

    


    
    #add tub to save data

    inputs=['cam/image_array',
            'user/angle', 'user/throttle', 
            'user/mode']

    types=['image_array',
           'float', 'float',
           'str']

    
    th = TubHandler(path=cfg.DATA_PATH)
    tub = th.new_tub_writer(inputs=inputs, types=types, user_meta=meta)
    V.add(tub, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')


    if type(ctr) is LocalWebController:
        print("You can now go to <your pi ip address>:8887 to drive your car.")
    
        ctr.set_tub(tub)
        
        if cfg.BUTTON_PRESS_NEW_TUB:
    
            def new_tub_dir():
                V.parts.pop()
                tub = th.new_tub_writer(inputs=inputs, types=types, user_meta=meta)
                V.add(tub, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')
                ctr.set_tub(tub)
    
            ctr.set_button_down_trigger('cross', new_tub_dir)
        ctr.print_controls()
    
    #run the vehicle for 20 seconds
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, 
            max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()
    
    if args['drive']:
        model_type = args['--type']
        camera_type = args['--camera']
        drive(cfg, model_path=args['--model'], use_joystick=args['--js'], model_type=model_type, camera_type=camera_type,
            meta=args['--meta'])
    
    if args['train']:
        from train import multi_train, preprocessFileList
        
        tub = args['--tub']
        model = args['--model']
        transfer = args['--transfer']
        model_type = args['--type']
        continuous = args['--continuous']
        aug = args['--aug']     

        dirs = preprocessFileList( args['--file'] )
        if tub is not None:
            tub_paths = [os.path.expanduser(n) for n in tub.split(',')]
            dirs.extend( tub_paths )

        multi_train(cfg, dirs, model, transfer, model_type, continuous, aug)

