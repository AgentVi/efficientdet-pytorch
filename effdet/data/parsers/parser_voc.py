""" Pascal VOC dataset parser

Copyright 2020 Ross Wightman
"""
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np

from .parser import Parser
from .parser_config import VocParserCfg

labelToGTClassMap = {
    "car":"innovi.vehicle.car", 
    "motorcycle":"innovi.twoWheeled.motorcycle", 
    "suv":"innovi.vehicle.car", 
    "bicycle":"innovi.twoWheeled.bicycle", 
    "personStanding":"innovi.people.personStanding", 
    "smallTruck":"innovi.vehicle.truck", 
    "bus":"innovi.vehicle.bus", 
    "schoolBus": "innovi.vehicle.bus",
    "miniBus": "innovi.vehicle.bus",
    "cityBus": "innovi.vehicle.bus",
    "personOnTheGround":"innovi.people.personOnTheGround", 
    "largeAnimal":"innovi.animal.largeAnimal",
    "personOverhead":"innovi.people.personStanding", 
    "van":"innovi.vehicle.van", 
    "pickupTruck":"innovi.vehicle.pickupTruck", 
    "mediumTruck":"innovi.vehicle.truck", 
    "personSitting":"innovi.people.personOnTheGround",
    "smallAnimal":"innovi.animal.smallAnimal",
    "bird":"innovi.animal.bird", 
    "bigTruck":"innovi.vehicle.truck",  
    "tractor":"innovi.vehicle.truck",
    "animal": "innovi.animal.largeAnimal",
    "large animal": "innovi.animal.largeAnimal",
    "large_animal": "innovi.animal.largeAnimal",
    "catDog": "innovi.animal.smallAnimal",
    "person": "innovi.people.personStanding",
    "person standing": "innovi.people.personStanding",
    "truck": "innovi.vehicle.truck",
    "big truck": "innovi.vehicle.truck",
    "big_truck": "innovi.vehicle.truck",
    "bigTruck":"innovi.vehicle.truck",
    "smallTruck": "innovi.vehicle.truck",
    "mediumTruck": "innovi.vehicle.truck",
    "vehicle": "innovi.vehicle.car",
    "suv": "innovi.vehicle.car",
    "vehicleNight": "innovi.vehicle.car",
    "person on the ground": "innovi.people.personOnTheGround",
    "person_on_the_ground": "innovi.people.personOnTheGround",
    "personOnTheGround": "innovi.people.personOnTheGround",
    "person overhead": "innovi.people.personStanding",
    "person_overhead": "innovi.people.personStanding",
    "pickup truck": "innovi.vehicle.pickupTruck",
    "pickup_truck": "innovi.vehicle.pickupTruck",
    "person-crouching": "innovi.people.personOnTheGround",
    "person-lying-down": "innovi.people.personOnTheGround",
    "personSitting": "innovi.people.personStanding",
    "person-sitting-floor": "innovi.people.personOnTheGround",
    "person-sitting-chair": "innovi.people.personStanding",
    "person-standing": "innovi.people.personStanding",
    "person_standing": "innovi.people.personStanding",
    "schoolBus":"innovi.vehicle.bus",
    "miniBus":"innovi.vehicle.bus",
    "cityBus":"innovi.vehicle.bus",
    "semiTrailers":"innovi.vehicle.truck",
    "forklift":"innovi.vehicle.truck",
    "compactCar":"innovi.vehicle.car"                             
}

class VocParser(Parser):

    DEFAULT_CLASSES = (
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, cfg: VocParserCfg):
        super().__init__(
            bbox_yxyx=cfg.bbox_yxyx,
            has_labels=cfg.has_labels,
            include_masks=False,  # FIXME to support someday
            include_bboxes_ignore=False,
            ignore_empty_gt=cfg.has_labels and cfg.ignore_empty_gt,
            min_img_size=cfg.min_img_size
        )
        self.correct_bbox = 1
        self.keep_difficult = cfg.keep_difficult

        self.anns = None
        self.img_id_to_idx = {}
        self._load_annotations(
            split_filename=cfg.split_filename,
            img_filename=cfg.img_filename,
            ann_filename=cfg.ann_filename,
            classes=cfg.classes,
        )

    def _load_annotations(
            self,
            split_filename: str,
            img_filename: str,
            ann_filename: str,
            classes=None,
    ):
        classes = classes or self.DEFAULT_CLASSES
        self.cat_names = list(classes)
        self.cat_ids = self.cat_names
        self.cat_id_to_label = {cat: i + self.label_offset for i, cat in enumerate(self.cat_ids)}

        self.anns = []

        with open(split_filename) as f:
            ids = f.readlines()
        for img_id in ids:
            img_id = img_id.strip("\n")
            filename = img_filename % img_id
            xml_path = ann_filename % img_id
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            if min(width, height) < self.min_img_size:
                continue

            anns = []
            for obj_idx, obj in enumerate(root.findall('object')):
                name = obj.find('name').text
                if name not in self.cat_id_to_label.keys():
                    name = labelToGTClassMap[name]
                label = self.cat_id_to_label[name]
                difficult = int(obj.find('difficult').text)
                bnd_box = obj.find('bndbox')
                bbox = [
                    int(float(bnd_box.find('xmin').text)),
                    int(float(bnd_box.find('ymin').text)),
                    int(float(bnd_box.find('xmax').text)),
                    int(float(bnd_box.find('ymax').text))
                ]
                anns.append(dict(label=label, bbox=bbox, difficult=difficult))

            if not self.ignore_empty_gt or len(anns):
                self.anns.append(anns)
                self.img_infos.append(dict(id=img_id, file_name=filename, width=width, height=height))
                self.img_ids.append(img_id)
            else:
                self.img_ids_invalid.append(img_id)

    def merge(self, other):
        assert len(self.cat_ids) == len(other.cat_ids)
        self.img_ids.extend(other.img_ids)
        self.img_infos.extend(other.img_infos)
        self.anns.extend(other.anns)

    def get_ann_info(self, idx):
        return self._parse_ann_info(self.anns[idx])

    def _parse_ann_info(self, ann_info):
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for ann in ann_info:
            ignore = False
            x1, y1, x2, y2 = ann['bbox']
            label = ann['label']
            w = x2 - x1
            h = y2 - y1
            if w < 1 or h < 1:
                ignore = True
            if self.yxyx:
                bbox = [y1, x1, y2, x2]
            else:
                bbox = ann['bbox']
            if ignore or (ann['difficult'] and not self.keep_difficult):
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)

        if not bboxes:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0, ), dtype=np.float32)
        else:
            bboxes = np.array(bboxes, ndmin=2, dtype=np.float32) - self.correct_bbox
            labels = np.array(labels, dtype=np.float32)

        if self.include_bboxes_ignore:
            if not bboxes_ignore:
                bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
                labels_ignore = np.zeros((0, ), dtype=np.float32)
            else:
                bboxes_ignore = np.array(bboxes_ignore, ndmin=2, dtype=np.float32) - self.correct_bbox
                labels_ignore = np.array(labels_ignore, dtype=np.float32)

        ann = dict(
            bbox=bboxes.astype(np.float32),
            cls=labels.astype(np.int64))

        if self.include_bboxes_ignore:
            ann.update(dict(
                bbox_ignore=bboxes_ignore.astype(np.float32),
                cls_ignore=labels_ignore.astype(np.int64)))
        return ann

