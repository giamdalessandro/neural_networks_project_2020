import os
import cv2
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET   # phone home.
from scipy.io import loadmat

ANNOTATIONS = "./dataset/raw_data/PascalVOC_2010_part/annotations_trainval/Annotations_Part/"


def read_part_annotations(img_name, anno_path=ANNOTATIONS):
    anno_file = os.path.join(anno_path, img_name[:-4] + ".mat")
    anno = loadmat(anno_file)
    categories = anno["anno"][0][0][1][0]     # matlab shit
    
    anno_dict = {}
    for c in categories:
        cat = c[0][0]
        if cat in anno_dict:
            anno_dict[cat]["bbox"].append(c[2][0])
            for p in list(c[3][0]):
                anno_dict[cat]["parts"].append(p)

        else:
            anno_dict[cat] = {
                "cat_id": c[1][0][0],
                "bbox"  : [c[2][0]],
                "parts" : list(c[3][0])
            }

    return anno_dict

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