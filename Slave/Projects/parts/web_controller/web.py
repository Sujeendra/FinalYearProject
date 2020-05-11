#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A web server used to control car using keys 
"""


import os
import json
import time
import asyncio

import requests
import tornado.ioloop
import tornado.web
import tornado.gen

from ... import utils


    
    
class LocalWebController(tornado.web.Application):

    def __init__(self):
        ''' 
        Create and publish variables needed on many of 
        the web handlers.
        '''

        print('Starting Industry Server...')

        this_dir = os.path.dirname(os.path.realpath(__file__))
        self.static_file_path = os.path.join(this_dir, 'templates', 'static')
        
        self.angle = 0.0
        self.throttle = 0.0
        self.mode = 'user'
        self.recording = False
        self.flag='None'
        handlers = [
            (r"/", tornado.web.RedirectHandler, dict(url="/drive")),
            (r"/drive", DriveAPI),
            (r"/video",VideoAPI),
            (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": self.static_file_path}),
            ]

        settings = {'debug': True}

        super().__init__(handlers, **settings)

    def update(self, port=8887):
        ''' Start the tornado webserver. '''
        #if self.flag=='user':
          #self.mode='local'

        asyncio.set_event_loop(asyncio.new_event_loop())
        print(port)
        self.port = int(port)
        self.listen(self.port)
        tornado.ioloop.IOLoop.instance().start()

    def run_threaded(self, img_arr=None):
        self.img_arr = img_arr
        #self.flag=flag       #if flag=='user':
       # self.application.mode='local'
        #print(type(self.application.mode))
        #print(flag)
        print("Local web controller working")  
        return self.angle, self.throttle, self.mode, self.recording
        
    def run(self, img_arr=None):
        self.img_arr = img_arr
        #self.flag=flag
        #if flag=='user':
          #self.mode='local'
        
       #self.img_arr = img_arr

        return self.angle, self.throttle,self.mode, self.recording

    def shutdown(self):
        pass


class DriveAPI(tornado.web.RequestHandler):

    def get(self):
        data = {}
        self.render("templates/vehicle.html", **data)
    
    
    def post(self):
        '''
        Receive post requests as user changes the angle
        and throttle of the vehicle on a the index webpage
        '''
        data = tornado.escape.json_decode(self.request.body)
        self.application.angle = data['angle']
        self.application.throttle = data['throttle']
        #if self.flag=='user':
           #self.application.mode='local'
        #else:
          
        self.application.mode = data['drive_mode']
        self.application.recording = data['recording']


class VideoAPI(tornado.web.RequestHandler):
    '''
    Serves a MJPEG of the images posted from the vehicle. 
    '''
    async def get(self):

        self.set_header("Content-type", "multipart/x-mixed-replace;boundary=--boundarydonotcross")

        self.served_image_timestamp = time.time()
        my_boundary = "--boundarydonotcross\n"
        while True:
            
            interval = .1
            if self.served_image_timestamp + interval < time.time():


                img = utils.arr_to_binary(self.application.img_arr)

                self.write(my_boundary)
                self.write("Content-type: image/jpeg\r\n")
                self.write("Content-length: %s\r\n\r\n" % len(img)) 
                self.write(img)
                self.served_image_timestamp = time.time()
                try:
                    await self.flush()
                except tornado.iostream.StreamClosedError:
                    pass
            else:
                await tornado.gen.sleep(interval)
