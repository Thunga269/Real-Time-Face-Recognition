import cv2
import datetime
import time
import config as cfg
from camera import ThreadedCamera

class Checkin(object):
    def __init__(self, face_model, sendlog):
        # self.IP_CAM = IP_CAM['IP_CAM']
        # self.protocol_url = 'rtsp://' + \
        #     IP_CAM['USER_CAM'] + ':' + IP_CAM['PASS_CAM'] + \
        #     '@' + self.IP_CAM + '/axis-media/media.amp'
        self.cam = ThreadedCamera()
        self.face_model = face_model
        self.frame = None
        self.sendlog = sendlog
        self.telegram_warning = cfg.config_sendlog['telegram_warning']
            
    def process_checkin(self):
        prev_frame_time = 0
        new_frame_time = 0
        it = 0
        while True:
            self.frame = self.cam.grab_frame()
            if self.frame is not None:
                new_frame_time = time.time()
                try: 
                    try:
                        recog_time_st = time.time()
                        res, dimg = self.face_model.face_recog(self.frame)
                        recog_time_en = time.time()
                        print("Recog time", recog_time_en - recog_time_st)
                        new_frame_time = time.time()
                        fps = 1/(new_frame_time-prev_frame_time)
                        prev_frame_time = new_frame_time  
                        fps = round(fps, 2)
                        fps = str(fps) 
                        cv2.putText(dimg, 'FPS: {}'.format(fps), (20,45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
                        cv2.imshow('CAMERA', dimg)   
                        if cv2.waitKey(1) == 27: # Esc
                            cv2.destroyAllWindows()
                            break 
                    except Exception as ex:
                        print('ERROR: Face Checkin | ', ex)             
                        if self.face_model.lock.locked():
                            self.face_model.lock.release()
                    message = ''
                    for per in res:
                        if per['name'] != 'unknown' and per['name'] != 'spoof_face':
                            message += per['name'] + ' '           
                    if message.strip() != '':
                        self.sendlog.send_telegram_message(
                            message + str(datetime.datetime.now().replace(microsecond=0)))
                      
                except Exception as ex:
                    print('ERROR: checkin file | ', ex)

            
