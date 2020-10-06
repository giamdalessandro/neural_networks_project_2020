import os
import cv2
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET   # phone home.


def read_content(img_name, anno_path):

    xml_file = anno_path + img_name + ".xml"
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    for boxes in root.iter('object'):
        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes


#d = read_part_annotations("2010_005993")
#print(d.keys())
#for i in range(356):
#    if "1" in str(d["bird"]["parts"][0][1][i]):
#        print(d["bird"]["parts"][0][1][i])