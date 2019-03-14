import numpy as np
import os
import sys
import cv2
from glob import glob
import tensorflow as tf
from tqdm import tqdm
from enum import Enum, auto

from matplotlib import pyplot as plt
sys.path.append("..")
from test import ObjectDetector, load_image_for_detection


if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


class PascalIndexes(Enum):
    background = 0
    aeroplane = auto()
    bicycle = auto()
    bird = auto()
    boat = auto()
    bottle = auto()
    bus = auto()
    car = auto()
    cat = auto()
    chair = auto()
    cow = auto()
    diningtable = auto()
    dog = auto()
    horse = auto()
    motorbike = auto()
    person = auto()
    pottedplant = auto()
    sheep = auto()
    sofa = auto()
    train = auto()
    tvmonitor = auto()

    @classmethod
    def get_member_list(cls):
        return [name for name, _ in cls.__members__.items() if name != "background"]


class PascalVOC(object):

    def __init__(self):
        self.dataset_root_path = "/mnt/poplin/share/dataset/PASCAL_VOC_2012"
        self.val_obj_class_list = [
            "aeroplane",
            "bicycle",
            "car",
            "chair",
            "motorbike",
            "sofa",
            "train",
        ]
        self.object_detector = ObjectDetector()

    def output_bbox(self, data_type="val", threshold=0.8):
        target_cls_list = PascalIndexes.get_member_list()
        output_root_dir = "/mnt/workspace2016/miyauchi/dataset/PascalBBox/%s" % data_type
        for obj_class in target_cls_list:
            os.makedirs(os.path.join(output_root_dir, obj_class), exist_ok=True)

        segmentation_image_paths = glob(
            "/mnt/poplin/share/dataset/PASCAL_VOC_2012/VOCdevkit/VOC2012/SegmentationClass_%s/*" % data_type)
        for image_path in tqdm(segmentation_image_paths):
            mask_image = cv2.imread(image_path, 0)
            file_name = os.path.basename(image_path)
            for obj_class in target_cls_list:
                mask = (mask_image == int(PascalIndexes[obj_class].value))
                if mask.any() == True:
                    original_image_path = os.path.join(self.dataset_root_path, "VOCdevkit/VOC2012/PNGImages/%s" % (file_name,))
                    image = load_image_for_detection(original_image_path)
                    output_path = os.path.join(output_root_dir, obj_class, file_name.replace("png", "txt"))
                    output_dict = self.object_detector.get_output_dict(image)
                    bbox_result = list()
                    for i in range(output_dict["num_detections"]):
                        if output_dict["detection_scores"][i] < threshold:
                            break
                        detection_class = self.object_detector.get_cls_name_by_index(output_dict["detection_classes"][i])

                        if detection_class == obj_class:
                            bbox_result.append(output_dict["detection_boxes"][i])
                    bbox_result = np.array(bbox_result)
                    np.savetxt(output_path, bbox_result, fmt="%.10f")


if __name__ == "__main__":
    pascal = PascalVOC()
    pascal.output_bbox()
