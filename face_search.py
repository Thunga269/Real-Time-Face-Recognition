import hnswlib
import numpy as np
import json

class FaceSearcher(object):
    def __init__(self, dim=512, space='cosine', threshold=0.4):
        self.p = hnswlib.Index(space=space, dim=dim)
        self.p.init_index(max_elements=7000, ef_construction=200, M=48)
        self.p.set_ef(50)
        self.k = 1
        self.threshold = threshold

    def add_faces(self, data, index):
        try:
            if index.shape[0] != data.shape[0]:
                print('Try to assign index with length {} to data with length {}'.format(
                    index.shape[0], data.shape[0]))
            else:
                self.p.add_items(data, index)
        except Exception as err:
            print("ERROR: FaceSearcher add_faces | ", err)

    def query_faces(self, data, names):
        try:
            index, distance = self.p.knn_query(data, k=self.k)
            index = np.squeeze(index)
            distance = np.squeeze(distance)
            print("index: ", index, "distance: ", distance)         
            if distance < self.threshold:
                info = names[index]
                if int(names[index]['face_mask']) == 1 and distance > self.threshold - 0.2:
                    info = json.loads('{"name": "unknown"}')  
            else:   
                info = json.loads('{"name": "unknown"}') 
            return info, distance
        except Exception as err:
            print("ERROR: query_faces | ", err)
            return None
    
    def save_index(self, path):
        self.p.save_index(path)
        return True

    def load_index(self, path, num_elements):
        self.p.load_index(path, num_elements)
        return True
        
