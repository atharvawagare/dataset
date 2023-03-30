import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from sort import *

class Detection:

    def __init__(self, source="inference/images", weights="yolov7.pt", img_size=640, conf_thres=0.25, iou_thres=0.45, device="", view_img=False, save_txt=False, save_conf=False, nosave=False, classes=80, agnostic_nms=False, augment=False, update=False, project="runs/detect", name="exp", exist_ok=False, no_trace=False, track=False):
        self.source=source
        self.weights=weights
        self.imgsz=img_size
        self.conf_thres=conf_thres
        self.iou_thres=iou_thres
        self.device=device
        self.view_img=view_img
        self.save_txt=save_txt
        self.save_conf=save_conf
        self.nosave=nosave
        self.classes=classes
        self.agnostic_nms=agnostic_nms
        self.augment=augment
        self.update=update
        self.project=project
        self.name=name
        self.exist_ok=exist_ok
        self.trace=not no_trace

        self.save_img=not nosave and not source.endswith('.txt')
        self.webcam=source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        
    def run(self):

        sort_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2) 

        with torch.no_grad():
            if self.update:  # update all models (to fix SourceChangeWarning)
                for self.weights in ['yolov7.pt']:
                    self.detect()
                    strip_optimizer(self.weights)
            else:
                self.detect()

"""Function to Draw Bounding boxes"""
    def draw_boxes(self, img, bbox, identities=None, categories=None, confidences = None, names=None, colors = None):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            tl = opt.thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

            cat = int(categories[i]) if categories is not None else 0
            id = int(identities[i]) if identities is not None else 0
            # conf = confidences[i] if confidences is not None else 0

            color = colors[cat]
            
            if not opt.nobbox:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

            if not opt.nolabel:
                label = str(id) + ":"+ names[cat] if identities is not None else  f'{names[cat]} {confidences[i]:.2f}'
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        return img

# I have removed save_img=False from detect function signature
    def detect(self):

        # Directories
        save_dir = Path(increment_path(Path(self.project) / self.name, self.exist_ok))  # increment run
        (save_dir / 'labels' if self.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        self.device = select_device(self.device)
        half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check img_size

        if self.trace:
            model = TracedModel(model, self.device, self.imgsz)

        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if self.webcam:
            self.view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=self.imgsz, stride=stride)
        else:
            dataset = LoadImages(self.source, img_size=self.imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if self.device.type != 'cpu':
            model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=self.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=self.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if self.webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Added --------
                    dets_to_sort = np.empty((0,6))

                    for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                        dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))

                    if self.track:
  
                        tracked_dets = sort_tracker.update(dets_to_sort, opt.unique_track_color)
                        tracks=sort_tracker.getTrackers()

                        # draw boxes for visualization
                        if len(tracked_dets)>0:
                            bbox_xyxy = tracked_dets[:,:4]
                            identities = tracked_dets[:, 8]
                            categories = tracked_dets[:, 4]
                            confidences = None

                    else:
                        bbox_xyxy = dets_to_sort[:,:4]
                        identities = None
                        categories = dets_to_sort[:, 5]
                        confidences = dets_to_sort[:, 4]

                    im0 = self.draw_boxes(im0, bbox_xyxy, identities, categories, confidences, names, colors)
                    # ---------------
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if self.save_img or self.view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Stream results
                if self.view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if self.save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        if self.save_txt or self.save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if self.save_txt else ''
            #print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')