import os
import os.path as osp
import cv2
import numpy as np
import warnings

from .src.anti_spoof_predict import AntiSpoofPredict
from .src.generate_patches import CropImage
from .src.utility import parse_model_name
warnings.filterwarnings('ignore')

class FASNetPredict(object):
    def __init__(self, model_dir, fasnetv2_path, fasnetv1se_path):
        self.device_id = 0
        self.this_dir = osp.dirname(__file__)
        self.model_dir = model_dir
        
        self.fasnetv2_path = fasnetv2_path
        self.fasnetv1se_path = fasnetv1se_path

        self.model_test = AntiSpoofPredict(self.device_id, fasnetv2_path = self.fasnetv2_path, fasnetv1se_path = self.fasnetv1se_path)
        self.image_cropper = CropImage()

    def check_image(self, image):
        height, width, channel = image.shape

        if width/height != 3/4:
            return False
        else:
            return True

    def predict(self, image, image_bbox=None):
        prediction = np.zeros((1, 3))
        
        for model_name in os.listdir(self.model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }

            if scale is None:
                param["crop"] = False
                
            if image_bbox is not None:
                img = self.image_cropper.crop(**param)

                prediction += self.model_test.predict(img, os.path.join(self.model_dir, model_name))
            else:
                img = cv2.resize(image, (w_input, h_input))
                
                prediction += self.model_test.predict(img, os.path.join(self.model_dir, model_name))
        
        label = np.argmax(prediction)

        value = prediction[0][label]/2
        
        if label == 1:
            return True  
        else:
            return False
