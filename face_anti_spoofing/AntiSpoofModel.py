from .FASNet.fasnet_predict import *
import config

class AntiSpoofModel(object):
    def __init__(self, config):
        self.model_fas = FASNetPredict(model_dir = config['model_fasnet_path'], 
                                            fasnetv2_path = config['model_fasnetv2_path'], 
                                            fasnetv1se_path = config['model_fasnetv2_path'])

    def fas_predict(self, img, img_bbox=None):   
        res = self.model_fas.predict(img, img_bbox)
        return res

