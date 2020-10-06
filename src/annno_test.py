


ANNOTATIONS = "./dataset/raw_data/PascalVOC_2010_part/VOCdevkit/VOC2010/Annotations/"


def read_part_annotations(img, anno_path):
    return NotImplementedError

def read_content(img, anno_path=ANNOTATIONS):
    xml_file = anno_path + img + ".xml"
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