import os
import csv
import cv2

import torch

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg,CfgNode
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer

from microfluidics_datasets import data_set_function_test, data_set_function_train, CustomTraniner

DatasetCatalog.register("microfluidics_train", data_set_function_train)
# MetadataCatalog.get("microfluidics_train").thing_classes(["CD8", "activated CD8", "platelets"])

def main():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/rpn_R_50_C4_1x.yaml"))
    cfg.DATASETS.TRAIN = ("microfluidics_train")
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 5
    cfg.SOLVER.BASE_LR = 0.000005
    cfg.SOLVER.MAX_ITER = 40000
    cfg.SOLVER.STEPS = []
    cfg.INPUT.FORMAT = "RGB"
    # cfg.INPUT.RANDOM_FLIP = "horizontal"
    # cfg.INPUT.CROP = CfgNode({"ENABLED": False})
    # cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3
    cfg.MODEL.RETINANET.NUM_CLASSES = 3

    # cfg.MODEL.RPN.NMS_THRESH = 0.5
    # cfg.MODEL.RPN.IOU_THRESHOLDS = [0.05, 0.95]
    # cfg.MODEL.RPN.IOU_LOSS_WEIGHT = 2
    # cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 10
    # cfg.MODEL.RPN.IOU_LOSS_WEIGHT = 2
    # cfg.MODEL.RETINANET.BBOX_REG_LOSS_TYPE = "giou"
    # cfg.MODEL.RETINANET.BBOX_REG

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CustomTraniner(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    f = open("{}/model_cfg.yaml".format(cfg.OUTPUT_DIR), 'w')
    f.write(cfg.dump())
    f.close()

if __name__ == '__main__':
    main()

pass