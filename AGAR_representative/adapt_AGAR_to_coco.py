__author__ = "Tom Fu"
__version__ = "1.0"

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import os
import json
from glob import glob
import cv2


def agar_to_coco_format(img_dir, img_format, shape="rectangle", destination_image_source_dir=None):
    current_img_id = 0
    output_l = []
    class_dict = {}
    class_num = 0

    for filename in glob(str(img_dir + '/*.' + img_format)):
        print(filename)
        current_json_name = "/".join(filename.split(
            '/')[:-1]) + "/" + filename.split('/')[-1].split('.')[0] + '.json'
        with open(current_json_name) as f:
            imgs_anns = json.load(f)

        current_img_dict = {}
        current_img_annotations = []

        # iterate thru each annotation
        for annotation in imgs_anns["labels"]:
            if annotation["class"] not in class_dict.values():
                class_dict.update({annotation["class"]: class_num})
                class_num += 1
            if shape == "rectangle":
                current_img_annotations.append({
                    'bbox': [annotation["x"], annotation["y"], annotation["x"] + annotation["width"], annotation["y"] + annotation["height"]],
                    'bbox_mode': BoxMode,
                    'category_id': class_dict[annotation["class"]],
                    'segmentation': [annotation["x"], annotation["y"], annotation["x"] + annotation["width"], annotation["y"] + annotation["height"]]
                })

        # update source image directory in case files are moved to a different directory than initially processed
        if destination_image_source_dir != None:
            current_img_filename = os.path.join(
                destination_image_source_dir, filename.split('/')[-1])
        else:
            current_img_filename = filename

        # generate outputs
        img = cv2.imread(filename)

        current_img_dict.update({'annotations': current_img_annotations,
                                 'file_name': current_img_filename,
                                 'height': img.shape[0],
                                 'image_id': current_img_id,
                                 'width': img.shape[1]})
        output_l.append(current_img_dict)
        current_img_id += 1

    print(output_l)
    return output_l


# img = cv2.imread(img)
agar_to_coco_format(
    '/Users/chenlianfu/Documents/Github/detectron2/AGAR_representative/lower-resolution', 'jpg')
