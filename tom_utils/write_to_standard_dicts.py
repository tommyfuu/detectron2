__author__ = "Tom Fu"
__version__ = "1.0"

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import os
import json


def get_dataset_dict(json_dir, json_name, destination_image_source_dir=None):
    'to convert the output json file into a dictionary digestable by detectron'

    # load json file
    json_file = os.path.join(json_dir, json_name)
    with open(json_file) as f:
        imgs_anns = json.load(f)

    # start making output list
    # format: each image is a dictionary, with keys annotation, file_name, image_id, height, width
    # annotation is a list of dictionaries, each of which contains keys bbox, bbox_mode, category_id, segmentation
    output_l = []
    img_l = imgs_anns['images']
    annotations_l = imgs_anns['annotations']
    # iterate thru each image
    for image in img_l:
        current_img_id = image['id']
        current_img_dict = {}
        current_img_annotations = []
        # iterate thru each annotation
        for annotation in annotations_l:
            if annotation['image_id'] == current_img_id:
                current_img_annotations.append({
                    'bbox': annotation['bbox'],
                    'bbox_mode': BoxMode,
                    'category_id': annotation['category_id'],
                    'segmentation': annotation['segmentation']
                })

        # update source image directory in case files are moved to a different directory than initially processed
        if destination_image_source_dir != None:
            current_img_filename = os.path.join(
                destination_image_source_dir, image['file_name'].split('/')[-1])
        else:
            current_img_filename = image['file_name']

        # generate outputs
        current_img_dict.update({'annotations': current_img_annotations,
                                 'file_name': current_img_filename,
                                 'height': image['height'],
                                 'image_id': current_img_id,
                                 'width': image['width']})
        output_l.append(current_img_dict)
    return output_l
