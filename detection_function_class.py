import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

class Detection:
    def __init__(self, save_txt=False, save_conf=False, device="", weights="yolov7.pt", img_size=640, trace=False, agnostic_nms=False, augment=False, conf_thres=0.25, iou_thres=0.45, classes=None):
        self.save_txt=save_txt
        self.save_conf=save_conf
        self.device=device
        self.weights=weights
        self.img_size=img_size
        self.trace=not trace
        self.agnostic_nms=agnostic_nms
        self.augment=augment
        self.conf_thres=conf_thres
        self.iou_thres=iou_thres
        self.classes=classes

        # Initialize
        set_logging()
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = check_img_size(self.img_size, s=self.stride)  # check img_size

        if self.trace:
            self.model = TracedModel(self.model, self.device, self.img_size)

        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if self.device.type != 'cpu':
            print("Inside Condition")
            print(self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters()))))
            print(type(self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))))
        self.old_img_w = self.old_img_h = self.img_size
        self.old_img_b = 1


    def detect(self, source):
        # Padded resize
        img = letterbox(source, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = numpy.ascontiguousarray(img)

        # for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            print("Inside Warmup")
            for i in range(3):
                self.model(img, augment=self.augment)[0]

        # Inference
        # t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            print("Inside Inference")
            print(img.shape)
            pred = self.model(img, augment=self.augment)[0]
        # t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        t3 = time_synchronized()

        for i, det in enumerate(pred):  # detections per image
            print("Inside Process Detections")
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], source.shape).round()
                print("Inside Rescaling")
        return det