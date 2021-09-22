import os
import csv
import cv2
import random

from detectron2.engine import DefaultPredictor
from detectron2.data import detection_utils as utils

from microfluidics_datasets import data_set_function_test
from microfluidics_datasets import data_set_function_train

import torch

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer

def main():

    # DatasetCatalog.register("microfluidics_train", data_set_function_train)
    # DatasetCatalog.register("microfluidics_test", data_set_function_test)

    cfg = get_cfg()
    f = open("{}/model_cfg.yaml".format("save_20210908"), 'r')
    cfg.load_cfg(f)
    # cfg.MODEL.META_ARCHITECTURE = 'GeneralizedRCNN'
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.99
    cfg.MODEL.RPN.NMS_THRESH = 0.99
    predictor = DefaultPredictor(cfg)

    data = data_set_function_train()

    for image_dict in data:
        im = utils.read_image(image_dict['file_name'], format='L')
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        outputs = predictor(im)

        v = Visualizer(im[:,:,::-1])
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("Test", out.get_image()[:,:,::-1])
        cv2.waitKey(0)

        pass

if __name__ == '__main__':
    main()