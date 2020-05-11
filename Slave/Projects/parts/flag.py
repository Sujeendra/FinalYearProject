"""
Flag used to turn on or off Localwebcontroller
"""
class Flag():
   def __init__(self):
      print("Flag part initialised")
   def run(self,cam_img_array):
      return 'local'
   def shutdown(self):
     pass
