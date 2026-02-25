from xml.etree import ElementTree
import os
import json
import numpy as np
from shapely.geometry import Polygon
import cv2


def load_vmeta(path):
    ns = "{http://ltsc.ieee.org/xsd/LOM}"
    tree = ElementTree.parse(path)
    root = tree.getroot()

    # Find video file name at technical/location
    technical = root.find("{}technical".format(ns))
    if technical is None:
        raise SyntaxError("Missing technical field in vmeta file.")
    location = technical.find("{}location".format(ns))
    if location is None:
        raise SyntaxError("Missing location attribute in technical field of vmeta file.")
    video_file = location.text

    # Find room cfg file and start time
    general = root.find("{}general".format(ns))
    if general is None:
        raise SyntaxError("Missing general field in vmeta file.")
    cfg_file = None
    start_time = None
    for identifier in general.findall("{}identifier".format(ns)):
        catalog = identifier.find("{}catalog".format(ns))
        entry = identifier.find("{}entry".format(ns))
        if catalog is not None and entry is not None:
            if catalog.text == "start_time":
                start_time = entry.text
            elif catalog.text == "space_metadata_file":
                cfg_file = entry.text
    if start_time is None:
        raise SyntaxError("Unable to find start_time in vmeta file.")
    if cfg_file is None:
        raise SyntaxError("Unable to find space_metadata_file in vmeta file.")
    
    # Format full paths based on vmeta path
    dir_name = os.path.dirname(path)
    cfg_file = os.path.join(dir_name, cfg_file)
    video_file = os.path.join(dir_name, video_file)
    return cfg_file, video_file, int(start_time)

def load_config(path):
    with open(path) as config_file:
        config = json.load(config_file)
    
    #mapping = np.array(config["Mapper"]).astype("int")
    boundary = np.array(config["Boundary"]).astype("int")
    pods = np.array(config["POD"]).astype("int")
    map_path = os.path.join(os.path.dirname(path), config["MapPath"])
    map_image = cv2.imread(map_path)

    #config["Mapper"] = PixelMapper(mapping[:,:2], mapping[:,2:])
    config["Boundary"] = Polygon(boundary)
    config["POD"] = pods
    config["Map Image"] = map_image

    return config