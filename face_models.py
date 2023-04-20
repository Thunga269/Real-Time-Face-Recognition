import numpy as np
from numpy.linalg import norm
import cv2
from tqdm import *
import face_analysis
import torch
import config
import math
import datetime
import config as cfg
import threading
from face_search import FaceSearcher
from face_anti_spoofing.AntiSpoofModel import AntiSpoofModel
from preprocess_input import preprocess_image, scale_coords, preprocess_image_fasnet
import time

class FaceModel(object):
    def __init__(self, model_fas = 'fasnet', sendlog = None):
        self.feature_path = cfg.feature_path
        self.info_path = cfg.info_path
        self.telegramid_path = cfg.telegramid_path
        
        try:
            self.names = list(np.load(self.info_path, allow_pickle=True))
        except Exception as ex:
            print(ex)
            self.names = []
            
        try:
            self.telegram_chatids = (np.load(self.telegramid_path, allow_pickle=True)).item()
        except Exception as ex:
            print(ex)
            self.telegram_chatids = {}

        self.face_model_recog = face_analysis.FaceAnalysis(name="antelopev2", root=".", 
                                                           allowed_modules = ['detection', 'recognition'],
                                                           providers=['CUDAExecutionProvider']
                                                          )
        self.face_model_recog.prepare(ctx_id=0, det_size=(640, 640))
        self.config_fas = config.config_fasnet

        self.model_fas = AntiSpoofModel(config = self.config_fas)

        self.model = FaceSearcher()
        self.model.load_index(self.feature_path, 7000)
        
        self.lock = threading.Lock()

        self.sendlog = sendlog
        self.log_checkin = []
        
        self.log_rereg = []
        
        self.check_sendmail = cfg.check_sendmail
        
        self.area_threshold = 13500

    def addNewFace(self, img, label):
        img = np.array(img)
        
        self.lock.acquire()
        faces = self.face_model_recog.get(img)
        self.lock.release()
        
        faces = sorted(faces, reverse=True, key = lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        face = faces[0]
        
        if (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]) < self.area_threshold/10:
            print('Add Face Failed')
            return False

        self.model.add_faces(np.array([face.embedding]),np.array([self.model.p.get_current_count()]))
        self.names.append(label.copy())
        self.model.save_index(self.feature_path)
        np.save(self.info_path, np.array(self.names)) 
        
        for i in label:
            if not label[i]:
                label[i] = 'NULL'

        print("Add Face Successfully")
        
        return True
    
    def face_recog(self, img):
        img_process = img
        
        dimg = img.copy()

        self.lock.acquire()
        bbox, landmark = self.face_model_recog.det_model.detect(img_process, max_num=0, metric='default')
        self.lock.release()
        
        res = []
        bboxes = []
        landmarks = []
        
        for i in range(len(bbox)):
            face_area = (bbox[i][2] - bbox[i][0]) * (bbox[i][3] - bbox[i][1])
            if face_area < self.area_threshold:
                res.append({'name': 'unknown'})
                continue
                
            input_fas = preprocess_image_fasnet(img_process, landmark[i])
            check = self.model_fas.fas_predict(input_fas)
            if check == False:
                cv2.rectangle(dimg, (int(bbox[i][0]), int(bbox[i][1])), (int(bbox[i][2]), int(bbox[i][3])), (0, 0, 255), 2)
                cv2.putText(dimg, 'spoof_face', (int(bbox[i][0]),int(bbox[i][1])-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                res.append({'name': 'spoof_face'})
            else:
                bboxes.append(bbox[i])
                landmarks.append(landmark[i])
        
        bboxes = np.array(bboxes)
        landmarks = np.array(landmarks)
        
        self.lock.acquire()
        faces = self.face_model_recog.get_feature(img_process, bboxes, landmarks)
        self.lock.release()
        
        print("Face counter:", len(faces))
        
        cv2.putText(dimg, "Face counter: %d" % len(faces), (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
        check_upload = False
        for face in faces:
            pred, prob = self.model.query_faces(face.embedding, self.names)
            print(pred, prob)
            bbox = face.bbox.astype(np.int)
            cv2.rectangle(dimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(dimg, pred['name'], (bbox[0],bbox[1]-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

            if pred != 'unknown':
                checkin_time = datetime.datetime.now().replace(microsecond=0)
                checkin_timestamp = str(int(datetime.datetime.timestamp(checkin_time)))
                
                data = {'info:employee_code': pred['employee_code'],
                        'info:name': pred['name'],
                        'info:email': pred['email'],
                        'info:checkin_time': checkin_timestamp,
                        }

                if self.check_sendmail == True:
                    t1 = time.time()
                    print('Time Check Log:', time.time() - t1)

                    if pred['email'] in self.telegram_chatids:
                        if self.telegram_chatids[pred['email']] is not None:
                            telegram_message = "📍 Hello Sir/Madam %s" % pred['name'] + \
                                "\n-------------------" + \
                                "\n👤 Employee code: %s" % pred['employee_code'] + \
                                "\n💌 Email: %s" % pred['email'] + \
                                "\n🏦 Full name %s" % pred['name'] + \
                                "\n⏰ Attendance time: 📅 %s 🕒 %s" % (str(checkin_time).split(' ')[0].strip(),
                                                                        str(checkin_time).split(' ')[1].strip()) + \
                                "\n Have a nice day!"

                            self.sendlog.send_telegram_message(telegram_message, {
                                'bot': cfg.config_sendlog['telegram_bot'],
                                'chat_id': self.telegram_chatids[pred['email']]
                            })
                    
                    self.log_checkin.append(
                        (data['info:email'], data['info:name'],
                            data['info:employee_code'],
                            str(datetime.datetime.now().replace(microsecond=0))))
                else:
                    self.log_checkin.append(
                        (data['info:email'], data['info:name'],
                         data['info:employee_code'],
                         str(datetime.datetime.now().replace(microsecond=0))))
            res.append(pred)
        return res, dimg
            
    def update_info(self, email, data):
        email = email.lower()
        try:
            telegram_chat_id = data['telegram_chat_id']
        except:
            telegram_chat_id = None
        
        if telegram_chat_id is not None:
            self.telegram_chatids[email] = telegram_chat_id
            np.save(self.telegramid_path, self.telegram_chatids)

        else:
            check_email = False
            for name in self.names:
                if name['email'] == email:
                    check_email = True
                    for key in data:
                        self.names[self.names.index(name)][key] = data[key]

            np.save(self.info_path, np.array(self.names))

            if not check_email:
                return False

        return True

    def get_info(self, email):
        for name in self.names:
            if name['email'] == email:

                info = {
                    'employee_code': name['employee_code'],
                    'name': name['name'],
                    'email': name['email'],
                    'face_mask': name['face_mask']
                }
                return info
        return

    def cal_similarity(self, data1, data2):
        pred = np.dot(data1, data2)/(norm(data1)*norm(data2))
        pred = min(pred, 1.0)
        pred = 1 - np.arccos(pred) / math.pi
        
        return str(round(pred * 100, 2)) + '%'

    def face_compare(self, img1, img2):
        img1 = preprocess_image(img1)[0]
        img2 = preprocess_image(img2)[0]
        
        # self.lock.acquire()
        faces1 = self.face_model_recog.get(img1)
        faces2 = self.face_model_recog.get(img2)
        # self.lock.release()
        
        try:
            faces1 = sorted(faces1, reverse=True, key = lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            face1 = faces1[0]

            faces2 = sorted(faces2, reverse=True, key = lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            face2 = faces2[0]
            
            sim = self.cal_similarity(face1.embedding, face2.embedding)

            return sim

        except:
            return "unknown"

    def face_check(self, img):
        img_process = preprocess_image(img)[0]
        faces, _ = self.face_model_recog.det_model.detect(img_process, max_num=0, metric='default')
        faces = sorted(faces, reverse=True, key = lambda x: (x[2] - x[0]) * (x[3] - x[1]))

        face = faces[0][:4]
        face = scale_coords(img_process.shape, torch.tensor([face.tolist()]), img.shape)
        face = face[0].tolist()
        
        face_bbox = [int(face[0]), int(face[1]), int(face[2]-face[0]+1), int(face[3]-face[1]+1)]
        
        res = self.model_fas.fas_predict(img, face_bbox)

        return res
