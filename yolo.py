#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a tensorflow2 saved_model converted from YOLO_v4 model
"""

from timeit import default_timer as timer  ### to calculate FPS
import cv2
import numpy as np
import tensorflow as tf


class YOLO(object):
    def __init__(self,config_file):
        self.model_path = './model_data/yolov4-coco/'
        self.classes_path = 'model_data/coco_classes.txt'
        self.score = 0.2  # 0.8
        self.iou = 0.1
        self.model_image_size = (416, 416)
        self.class_names = self.read_class_names(self.classes_path)
        self.detection_list = config_file["classes_list"]
        # self.model_image_size = (416, 416) # fixed size or (None, None) small targets:(320,320) mid targets:(960,960)
        self.is_fixed_size = self.model_image_size != (None, None)
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        except:
            print("GPU not found started with CPU")

        try:
            self.saved_model_loaded = tf.saved_model.load(self.model_path)
        except Exception as exp:
            print("Exception while loading the graphs for the person detection" + str(exp))

    def read_class_names(self, class_file_name):
        names = {}
        with open(class_file_name, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
            return names

    @tf.function
    def PPE_detecting_fn(self, inp):
        try:
            PPE_model_fn = self.saved_model_loaded.signatures['serving_default']
            res = PPE_model_fn(inp)
            return res
        except Exception as exp:
            print("Error loading model" + str(exp))

    def perform_PPE_detection(self, frame_expanded):
        frame = cv2.cvtColor(frame_expanded, cv2.COLOR_BGR2RGB)
        image_data = cv2.resize(frame, (416, 416))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        batch_data = tf.constant(image_data)
        pred_bbox = self.PPE_detecting_fn(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.25
        )

        return boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()

    def detect_image(self, frame):

        start_time = timer()
        frame_copy = np.copy(frame)

        boxes, scores, classes, valid_detections = self.perform_PPE_detection(frame_copy)

        box = np.squeeze(boxes)
        cls = np.squeeze(classes)
        num = np.squeeze((valid_detections))
        sc = np.squeeze(scores)
        image_h, image_w, _ = frame.shape

        box = box[0:int(num)]
        cls = cls[0:int(num)]
        sc = sc[0:int(num)]

        deleted_indx = []
        for i in range(box.shape[0]):

            if self.class_names[cls[i]] in self.detection_list:
                continue

            else:
                deleted_indx.append(i)

        out_boxes = np.delete(box, deleted_indx, axis=0)
        out_classes = np.delete(cls, deleted_indx, axis=0)
        out_scores = np.delete(sc, deleted_indx, axis=0)

        return_boxs = []
        return_class_name = []
        (h, w) = frame.shape[:2]
        if len(out_boxes) > 0:

            for i, c in reversed(list(enumerate(out_classes))):
                out_box = out_boxes[i]
                score = round(out_scores[i], 2)
                if score > self.score:
                    coor = out_box
                    coor[0] = int(coor[0] * image_h)
                    coor[2] = int(coor[2] * image_h)
                    coor[1] = int(coor[1] * image_w)
                    coor[3] = int(coor[3] * image_w)
                    startY = int(coor[0])
                    startX = int(coor[1])
                    endY = int(coor[2])
                    endX = int(coor[3])
                    w = int(coor[3]) - int(coor[1])
                    h = int(coor[2]) - int(coor[0])
                    return_boxs.append([startX, startY, w, h])
                    return_class_name.append(c)

        end_time = timer()
        print('*** Processing time for prediction: {:.2f}ms'.format((end_time - start_time) * 1000))
        return return_boxs, return_class_name
