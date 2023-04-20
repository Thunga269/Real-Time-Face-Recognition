#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import _init_path
import numpy as np
from typing import Optional
from flask import Flask, request, jsonify
import time, requests
from PIL import Image
from io import BytesIO
from pydantic.main import BaseModel
import io
import datetime
import cv2
import base64
import json
import threading
import tool.utils as utils
import config as cfg
from face_models import * 
from checkin import Checkin
from sendlog import SendLog

class Response(BaseModel):
    url: Optional[str] = None
    label: Optional[str] = None
    message: Optional[str] = None

sendlog = SendLog(cfg.config_sendlog)

face_model = FaceModel(sendlog=sendlog)

def download_image(image_url, is_arr = True):
    header = utils.random_headers()

    response = requests.get(image_url, headers=header, stream=True, verify=False, timeout=5)

    image = Image.open(BytesIO(response.content)).convert('RGB')
    
    if is_arr:
        image = np.array(image)

    return image

app = Flask(__name__)

@app.route("/")
def home():
    return "FaceID System"

@app.route("/faceid/predict", methods = ['POST', 'GET'])
def face_predict():
    global face_model
    if request.method == 'GET':
        url = request.args.get('url', default='', type=str)

        img = download_image(url)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        res = face_model.face_recog(img)
        
        if res == 'unknown':
            js = {'label':'Không nhận diện được', 'status':False}
        else:
            js = {'label':'Nhận diện ra ' + str(res), 'status':True}
        
        return json.dumps(js,ensure_ascii=False).encode('utf8')
    
    elif request.method == 'POST':
        img = request.files.to_dict()['image'].read()

        img = Image.open(io.BytesIO(img)).convert('RGB')
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        res = face_model.face_recog(img)
        
        pred = ''
        for re in res:
            if re == 'unknown':
                continue
            pred += re + ', '
        
        num_unknown = res.count('unknown')
        
        if num_unknown != 0:
            pred += str(num_unknown) + ' unknown'
        pred = pred.strip(', ')
        
        if pred == '':
            js = {'label':'Không nhận diện được', 'status':False}
        else:
            js = {'label':'Nhận diện ra ' + str(pred), 'status':True}
        
        return json.dumps(js,ensure_ascii=False).encode('utf8')

@app.route("/faceid/register", methods = ['POST'])
def register():
    global face_model
    img = request.files.to_dict()['image'].read()
    label = {
        'name': request.form['name'],
        'email': request.form['email'],
        'employee_code': request.form['employee_code'],
        'face_mask': request.form['face_mask']
    }
    img = Image.open(io.BytesIO(img)).convert('RGB')
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if int(label['face_mask']) == 0:
        face_mask = 'Không có khẩu trang'
    else:
        face_mask = 'Có khẩu trang'
    utils.save_image(img, face_mask + ' - ' + label['name'] + ' - ' + label['email'] + ' - ' + label['employee_code'])
    try:
        add_result = face_model.addNewFace(img, label)
        if add_result == True:
            js = {'msg':'Đăng ký khuôn mặt thành công', 'status': True}
            return json.dumps(js,ensure_ascii=False).encode('utf8')
        else:
            js = {'msg':'Đăng ký khuôn mặt không thành công', 'status': False}
            return json.dumps(js,ensure_ascii=False).encode('utf8')           
    
    except Exception as e:
        js = {'msg':'Đăng ký khuôn mặt không thành công', 'status': False}
        print("ERROR: Add new face | ", e)
        return json.dumps(js,ensure_ascii=False).encode('utf8')


@app.route("/faceid/fas_predict", methods = ['POST'])
def face_anti_spoof_predict():
    global face_model
    img = request.files.to_dict()['image'].read()

    img = Image.open(io.BytesIO(img)).convert('RGB')
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    res = face_model.face_check(img)
        
    if res == True:
        js = {'label':'Khuôn mặt thật', 'status':True}
    else:
        js = {'label':'Khuôn mặt giả mạo', 'status':False}
        
    return json.dumps(js,ensure_ascii=False).encode('utf8')

@app.route("/faceid/update_info", methods = ['GET', 'POST'])
def update_info():
    global face_model
    
    try:
        email = request.json['email']
        data = request.json
        data.pop('email')

        res = face_model.update_info(email, data)
    except:
        js = {
            'status': 1,
            'code': 400,
            'message': 'FaceID System'
        }
        
        return jsonify(js)

    if res == True:
        js = {
            'status': 1,
            'code': 200,
            'message': 'Cập nhật thông tin thành công'
        }
    else:
        js = {
            'status': 1,
            'code': 400,
            'message': 'Cập nhật thông tin không thành công. Không tìm thấy địa chỉ email'
        }
        
    return jsonify(js)

@app.route("/faceid/get_info", methods = ['POST'])
def get_info():
    global face_model
    
    email = request.form['email']

    info = face_model.get_info(email)

    return info

@app.route("/faceid/compare", methods = ['POST'])
def compare():
    img1 = request.files.to_dict()['image1'].read()
    img2 = request.files.to_dict()['image2'].read()

    img1 = Image.open(io.BytesIO(img1)).convert('RGB')
    img1 = np.uint8(img1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    
    img2 = Image.open(io.BytesIO(img2)).convert('RGB')
    img2 = np.uint8(img2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    
    pred = face_model.face_compare(img1, img2)
    
    js = {'similarity':pred}
    return js
    
@app.route("/faceid/show_checkin_history", methods = ['POST'])
def show_checkin_history():
    try:
        num_record = int(request.form['number_record'])
        
        if num_record < 3:
            js = {
                'status': 1,
                'code': 400,
                'message': 'number_record must be >= 1'
            }

            return js
    except:
        js = {
            'status': 1,
            'code': 400,
            'message': 'FaceID System'
        }
        
        return js
    
    checkin_history = []
    for item in result:
        js_item = json.loads(json.dumps({key.decode(): val.decode() for key, val in item[1].items()}))
        js_item['info:image_id'] = js_item['info:link'].split('/')[-1].replace('_face_image.jpeg', '')
        
        checkin_history.append(js_item)
    
    js = {
        'status': 1,
        'code': 200,
        'message': 'FaceID System',
        'data': checkin_history
    }
    
    del result
    
    return js

@app.route("/faceid/show_face_image", methods = ['GET', 'POST'])
def show_face_image():
    global face_model
    
    if request.method == 'GET':
        image_id = request.args.get('image_id', default='', type=str)
    
    elif request.method == 'POST':
        image_id = request.form['image_id']
    
    try:
        image = download_image('https://pega-news.mediacdn.vn/%s_face_image.jpeg' % image_id, False)
    except:
        js = {
            'status': 0,
            'code': 200,
            'message': 'FaceID System'
        }

        return js
    
    b = io.BytesIO()
    image.save(b, 'jpeg')
          
    js = {
        'status': 1,
        'code': 200,
        'message': 'FaceID System',
        'data': (base64.b64encode(b.getvalue())).decode("utf-8") 
    }
       
    return js

def checkin():
    global face_model, sendlog
    checkin = Checkin(face_model, sendlog)
    checkin.process_checkin()

def thread_sendmail():
    global face_model
    
    while True:
        if face_model.log_checkin:
            data = face_model.log_checkin.pop(0)
            sendlog.send_email(data[0], data[1], data[2], data[3])
                
        time.sleep(1)
        
def api():
    app.run(utils.get_ip(), cfg.port, threaded=True, debug=False)
    
def main():
    t1 = threading.Thread(target=api)
    t2 = threading.Thread(target=checkin)
    t3 = threading.Thread(target=thread_sendmail)
    
    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

main()
