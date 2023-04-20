import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

lib_path = this_dir + "/face_anti_spoofing"
add_path(lib_path)

lib_path = this_dir + "/face_anti_spoofing/AENet"
add_path(lib_path)

lib_path = this_dir + "/face_anti_spoofing/FASNet"
add_path(lib_path)

lib_path = this_dir + "/tool"
add_path(lib_path)
