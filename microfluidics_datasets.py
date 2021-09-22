import copy
import os
import csv
import cv2
import torch
from detectron2.engine import DefaultTrainer
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.utils.visualizer import Visualizer

categories = {
    'CD8': 0,
    'activated CD8': 1,
    'platelet': 2
}

def data_set_function_test():
    test_images_path = os.path.join('test_images')
    # test_images_path = os.path.join('test_images')

    annotate_path = os.path.join('annotate-2class.txt')

    csv_file = csv.reader(open(annotate_path, 'r'))

    list_dict = []

    int_iter = 0

    image_dict = {}



    for row in csv_file:
        if row[0].split('/')[0] == 'test_images':
            dict_to_add = [
                {"bbox": [float(row[1]), float(row[2]), float(row[3]), float(row[4])]},
                {"bbox_mode": 0},
                {"category_id": categories[row[5]]},
            ]

            if row[0] in image_dict:
                image_dict[row[0]].append(dict_to_add)
            else:
                image_dict[row[0]] = [dict_to_add]

    for file in image_dict:
        image = cv2.imread(file)

        final_dict = {}

        final_dict['file_name'] = file
        final_dict['height'] = image.shape[0]
        final_dict['width'] = image.shape[1]
        final_dict['image_id'] = int_iter
        final_dict['annotations'] = image_dict[file]

        list_dict.append(final_dict)

        int_iter += 1

    return list_dict

def data_set_function_train():
    train_images_path = os.path.join('train_images')
    # test_images_path = os.path.join('test_images')

    annotate_path = os.path.join('annotate-2class.txt')

    csv_file = csv.reader(open(annotate_path, 'r'))

    list_dict = []

    int_iter = 0

    image_dict = {}

    categories = {
        'CD8': 0,
        'activated CD8': 1,
        'platelet': 2
    }

    for row in csv_file:
        if row[0].split('/')[0] == 'train_images':
            dict_to_add = {
                "bbox": [float(row[1]), float(row[2]), float(row[3]), float(row[4])],
                "bbox_mode": 0,
                "category_id": categories[row[5]]
            }

            if row[0] in image_dict:
                image_dict[row[0]].append(dict_to_add)
            else:
                image_dict[row[0]] = [dict_to_add]

    for file in image_dict:
        image = cv2.imread(file)

        final_dict = {}

        final_dict['file_name'] = file
        final_dict['height'] = image.shape[0]
        final_dict['width'] = image.shape[1]
        final_dict['image_id'] = int_iter
        final_dict['annotations'] = image_dict[file]

        list_dict.append(final_dict)

        int_iter += 1

    return list_dict

def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format='L')
    transform_list = [
        T.RandomBrightness(0.5, 1.5),
        T.RandomContrast(0.5, 1.5),
        T.RandomRotation(angle=[90, 90]),
        T.RandomFlip(prob=0.4, horizontal=True, vertical=False),
    ]

    image, transforms = T.apply_transform_gens(transform_list, image)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("uint8"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]

    dataset_dict["annotations"] = annos

    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    # v = Visualizer(image[:,:,::-1])
    # out = v.draw_dataset_dict(dataset_dict)
    # cv2.imshow("Test", out.get_image()[:,:,::-1])
    # cv2.waitKey(0)

    return dataset_dict

class CustomTraniner(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

def test_dataset():
    # Test train dataset
    data = data_set_function_train()

    data_0 = data[0]

    im = cv2.imread(data_0['file_name'])

    for i in data[0]['annotations']:
        points = i['bbox']
        pt1 = (int(points[0]), int(points[1]))
        pt2 = (int(points[2]), int(points[3]))
        color = (255, 255, 255)
        cv2.rectangle(im, pt1, pt2, color)

    # cv2.imshow("Result", im)
    # cv2.waitKey(0)