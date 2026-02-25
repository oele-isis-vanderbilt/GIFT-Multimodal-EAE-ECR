# @Author :  NaveedMohammed
# @File   :  vmeta.py

import xml.etree.ElementTree as gfg
import time
import os


def generate_vmeta(video_directory, video_basename, type):
    """

    Args:
        video_basename: filename of the video as defined in the original video vmeta file in the course folder
        video_directory: Output location for the videos and vmeta files (Domain session folder)
        type: label for a video file based on the type of video.
                ex: _MapTracking, _Tracking
    Returns:
        writes a vmeta xml file to the provided location

    """
    video_filename = video_basename + type + ".mp4"
    root = gfg.Element("lom")
    root.set("xmlns", "http://ltsc.ieee.org/xsd/LOM")

    e1 = gfg.Element("technical")
    root.append(e1)

    e1c1 = gfg.SubElement(e1, "location")
    e1c1.text = video_filename

    e2 = gfg.Element("general")
    root.append(e2)

    e2c1 = gfg.SubElement(e2, "identifier")
    e2c1d1 = gfg.SubElement(e2c1, "catalog")
    e2c1d1.text = "start_time"
    e2c1d2 = gfg.SubElement(e2c1, "entry")
    e2c1d2.text = str(int(time.time() * 1000))  # unix timestamp in milliseconds.

    e2c2 = gfg.SubElement(e2, "identifier")
    e2c2d1 = gfg.SubElement(e2c2, "catalog")
    e2c2d1.text = "offset"
    e2c2d2 = gfg.SubElement(e2c2, "entry")
    e2c2d2.text = "0"

    e2c3 = gfg.SubElement(e2, "identifier")
    e2c3d1 = gfg.SubElement(e2c3, "catalog")
    e2c3d1.text = "title"
    e2c3d2 = gfg.SubElement(e2c3, "entry")
    e2c3d2.text = video_filename

    tree = gfg.ElementTree(root)

    vmeta_filename = video_basename + type + ".vmeta.xml"
    vmeta_file_path = os.path.join(video_directory, vmeta_filename)
    with open(vmeta_file_path, "wb") as f:
        tree.write(f, encoding="UTF-8", xml_declaration=True)
