# needed
# pip install pyyaml==5.1
# pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

# check pytorch installation:
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer
from detectron2.data.catalog import Metadata
from AGAR_representative.adapt_AGAR_to_coco import agar_to_coco_format
import random
import json
import os
from AGAR_representative.adapt_AGAR_to_coco import agar_to_coco_format
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import cv2
import numpy as np
from detectron2.utils.logger import setup_logger
import detectron2
import torch
import torchvision
print(torch.__version__, torch.cuda.is_available())
# assert torch.__version__.startswith("1.9")

# Some basic setup:
# Setup detectron2 logger
setup_logger()

# import some common libraries
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
# for the line above, the following is required
# pip install torchvision==0.9.1

# wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg
im = cv2.imread("/Users/chenlianfu/Documents/Github/detectron2/researchSU19-187+cropped.jpeg")
# cv2_imshow(im)

# create a detectron2 config and a detectron2 DefaultPredictor to run inference on this image
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = 'cpu'
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)


# source_dir_path = '/Users/chenlianfu/Documents/Github/detectron2/AGAR_representative/lower-resolution'
# output_l, class_dict = agar_to_coco_format(
#     source_dir_path, 'jpg')
agar_metadata = Metadata()
for d in ["higher-resolution/bright", "higher-resolution/dark", "higher-resolution/vague", "lower-resolution"]:
    DatasetCatalog.register("agar_" + d, lambda d=d: agar_to_coco_format(
        "./AGAR_representative/" + d, 'jpg')[0])
    MetadataCatalog.get("agar_" + d).set(thing_classes=list(agar_to_coco_format(
        "/Users/chenlianfu/Documents/Github/detectron2/AGAR_representative/" + d, 'jpg')[1].keys()))

agar_metadata = MetadataCatalog.get("lower-resolution")


# attempt to print a training data
# dataset_dicts = agar_to_coco_format("./AGAR_representative/higher-resolution/bright", 'jpg')[0]
# print(dataset_dicts)
# d = dataset_dicts[2]
# img = cv2.imread(d["file_name"])
# visualizer = Visualizer(img[:, :, ::-1], metadata=agar_metadata, scale=0.5)
# print(visualizer)
# print(d)
# out = visualizer.draw_dataset_dict(d)
# print(out.get_image().shape)
# print(out.get_image())
# print("AAAAAH")
# print(out.get_image()[:, :, ::-1].shape)
# print(out.get_image()[:, :, ::-1])
# # cv2.imshow('', out.get_image()[:, :, ::-1])
# cv2.imwrite('./testing_training_new.jpg', out.get_image()[:, :, ::-1])


# try training
print("Start training", "aha")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("agar_higher-resolution/bright", "agar_higher-resolution/vague")
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.MAX_ITER = 300
cfg.SOLVER.STEPS = []        # do not decay learning rate
# faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
# only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.DEVICE = 'cpu'
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
print("check point", "aha")
trainer.resume_or_load(resume=False)
trainer.train()


# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
# path to the model we just trained
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)


dataset_dicts = agar_to_coco_format("./AGAR_representative/higher-resolution/dark", 'jpg')[0]
num = 1
for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=agar_metadata,
                   scale=0.5,
                   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   instance_mode=ColorMode.IMAGE_BW
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite('./testing' + str(num) + '.jpg', out.get_image()[:, :, ::-1])
