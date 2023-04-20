import cv2
from threading import Thread

class ThreadedCamera(object):
    def __init__(self, source = 0):

        self.capture = cv2.VideoCapture(source)
        # self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        
        self.thread = Thread(target = self.update, args = ())
        self.thread.daemon = True
        self.thread.start()

        self.status = False
        self.frame  = None

    def update(self):
        while True:
            if self.capture.isOpened():   
                (self.status, self.frame) = self.capture.read()

    def grab_frame(self):
        # print(self.fps)
        if self.status:
            return self.frame
        return None  
