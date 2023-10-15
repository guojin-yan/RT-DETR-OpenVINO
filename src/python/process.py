# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/14  14:42:39
# @Author  : Yan Guojin
# @File    : process.py
# @E-mail	: guojin_yjs@cumt.edu.cn
# @GitHub	: https://github.com/guojin-yan
# @Description : 

import cv2 as cv
import numpy as np
import PIL
from PIL import ImageDraw, Image

def print_info(msg):
    print("[INFO]  %s"%msg,end="\n")

# The `RtdetrProcess` class is a Python class that provides methods for preprocessing and
# postprocessing images for the RT-DETR object detection model.
class RtdetrProcess(object):
    def __init__(self, target_size, label_path=None, threshold=0.5, interp=cv.INTER_LINEAR):
        """
        This function initializes an object with target size, label path, threshold, and interpolation
        method.
        
        Args:
          target_size: The target size is the desired size of the image after resizing. It is usually
        specified as a tuple of width and height, such as (224, 224).
          label_path: The path to the file containing the labels for the data.
          threshold: The threshold parameter is used to determine the minimum confidence score required for
        an object detection to be considered valid. Any detection with a confidence score below the
        threshold will be ignored. The default value is 0.5, but you can change it to a different value if
        desired.
          interp: The "interp" parameter is used to specify the interpolation method to be used when
        resizing the images. In this case, the default interpolation method is set to "cv.INTER_LINEAR",
        which stands for bilinear interpolation. Bilinear interpolation is a commonly used method for
        resizing images, as it provides a smooth
        """
        self.im_info = dict()
        self.target_size =target_size
        self.interp = interp
        self.threshold = threshold
        if label_path is None:
            self.labels = []
            self.flabel = False
        else:
            self.labels = self.read_lable(label_path=label_path)
            self.flabel = True

    def read_lable(self,label_path):
        """
        The function `read_label` reads a file at the given `label_path` and returns a list of labels.
        
        Args:
          label_path: The `label_path` parameter is the path to the file containing the labels.
        
        Returns:
          the list of labels read from the label_path file.
        """
        lable = [] 
        f = open(label_path)
        line = f.readline()
        while line:
            lable.append(line.replace('\n',''))
            line = f.readline()
        f.close()
        return lable
    def preprocess(self,im):
        """
        The `preprocess` function resizes an image, normalizes its pixel values, and returns the processed
        image along with information about its shape and scale factor.
        
        Args:
          im: The input image that needs to be preprocessed.
        
        Returns:
          two values: `np.expand_dims(out_im.astype('float32'),0)` and `self.im_info`.
        """
        assert len(self.target_size) == 2
        assert self.target_size[0] > 0 and self.target_size[1] > 0
        origin_shape = im.shape[:2]
        resize_h, resize_w = self.target_size
        im_scale_y = resize_h / float(origin_shape[0])
        im_scale_x = resize_w / float(origin_shape[1])
        out_im = cv.resize(
            im.astype('float32'),
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)
        self.im_info['im_shape'] = np.array(im.shape[:2]).astype('float32')
        self.im_info['scale_factor'] = np.array([im_scale_y, im_scale_x]).astype('float32')
        scale = 1.0 / 255.0
        out_im *= scale
        out_im = out_im.transpose((2, 0, 1)).copy()

        return  np.expand_dims(out_im.astype('float32'),0), self.im_info
    
    def sigmoid(self,z):
        """
        The sigmoid function takes in an array of numbers and returns an array of their corresponding
        sigmoid values.
        
        Args:
          z: The parameter `z` is a list or array of numbers.
        
        Returns:
          a numpy array containing the sigmoid function applied to each element in the input array `z`.
        """
        fz = []
        for num in z:
            fz.append(1/(1+np.exp(-num)))
        return np.array(fz)
    def postprocess(self,scores,bboxs=None):
        """
        The `postprocess` function takes in scores and bounding boxes, and returns a list of dictionaries
        containing the class ID, label, score, and adjusted bounding box coordinates.
        
        Args:
          scores: The `scores` parameter is a list or array containing the predicted scores for each
        class. Each element in the list represents the scores for a single class. The scores should be in
        the range [0, 1].
          bboxs: The parameter `bboxs` is a list of bounding boxes. Each bounding box is represented as a
        list of four values: [x1, y1, x2, y2], where (x1, y1) are the coordinates of the top-left corner
        and (x2, y
        
        Returns:
          The function `postprocess` returns a list of dictionaries. Each dictionary represents a result
        and contains the following keys: "clsid" (class ID), "label" (class label), "score" (confidence
        score), and "bbox" (bounding box coordinates).
        """
        results = []
        if bboxs is None:
            scores = np.array(scores).astype('float32')
            for l in scores:
                if(l[1]>=self.threshold):
                    re = dict()
                    re["clsid"]=int(l[0])
                    if(self.flabel):
                        re["label"]=self.labels[int(l[0])]
                    else:
                        re["label"]=int(l[0])
                    re["score"]=l[1]
                    bbox=[l[2],l[3],l[4],l[5]]
                    re["bbox"]=bbox
                    results.append(re)
        else:
            scores = np.array(scores).astype('float32')
            bboxs = np.array(bboxs).astype('float32')
            for s,b in zip(scores,bboxs):
                s = self.sigmoid(s)
                if(np.max(np.array(s)>=self.threshold)):
                    ids = np.argmax(np.array(s))
                    re = dict()
                    re["clsid"]=int(ids)
                    if(self.flabel):
                        re["label"]=self.labels[int(ids)]
                    else:
                        re["label"]=int(ids)
                    re["score"]=s[ids]
                    cx=(b[0]*640.0)/self.im_info["scale_factor"][1]
                    cy=(b[1]*640.0)/self.im_info["scale_factor"][0]
                    w=(b[2]*640.0)/self.im_info["scale_factor"][1]
                    h=(b[3]*640.0)/self.im_info["scale_factor"][0]

                    bbox=[cx-w/2.0,
                          cy-h/2.0,
                          cx+w/2.0,
                          cy+h/2.0]
                    re["bbox"]=bbox
                    results.append(re)
        return results
    def get_color_map_list(self, num_classes):
        """
        The function `get_color_map_list` generates a color map list based on the number of classes
        provided.
        
        Args:
          num_classes: The parameter `num_classes` represents the number of classes or categories for which
        you want to generate a color map.
        
        Returns:
          a color map list.
        """
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
        return color_map
    def imagedraw_textsize_c(self, draw, text):
        """
        The function `imagedraw_textsize_c` calculates the width and height of a given text when drawn on an
        image using the `draw` object.
        
        Args:
          draw: The "draw" parameter is an instance of the `ImageDraw` class from the PIL library. It is
        used to draw on an image.
          text: The "text" parameter is the string of text that you want to measure the size of.
        
        Returns:
          the width (tw) and height (th) of the text when it is drawn on an image.
        """
        if int(PIL.__version__.split('.')[0]) < 10:
            tw, th = draw.textsize(text)
        else:
            left, top, right, bottom = draw.textbbox((0, 0), text)
            tw, th = right - left, bottom - top

        return tw, th
    def draw_box(self, im, results):
        """
        The `draw_box` function takes an image and a list of bounding box results, and draws the bounding
        boxes and labels on the image.
        
        Args:
          im: The parameter "im" is a numpy object. It represents the input image on which the bounding
        boxes will be drawn.
          results: results is a list of dictionaries, where each dictionary represents a detected object.
        Each dictionary has the following keys: `clsid`, `label`, `bbox`, `score`.
        
        Returns:
          a NumPy array representation of the visualized image.
        """
        draw_thickness = 2
        im = Image.fromarray(im)
        draw = ImageDraw.Draw(im)
        clsid2color = {}
        color_list = self.get_color_map_list(len(self.labels))
        for re in results:
            clsid =  re["clsid"]
            label = re["label"]
            bbox = re["bbox"]
            score = re["score"]
            if clsid not in clsid2color:
                clsid2color[clsid] = color_list[clsid]
            color = tuple(clsid2color[clsid])

            if len(bbox) == 4:
                xmin, ymin, xmax, ymax = bbox
                print_info('class_id:{:d}, label:{:s}, confidence:{:.4f}, left_top:[{:.2f},{:.2f}],'
                    'right_bottom:[{:.2f},{:.2f}]'.format(
                        int(clsid), label, score, xmin, ymin, xmax, ymax))
                # draw bbox
                draw.line(
                    [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                    (xmin, ymin)],
                    width=draw_thickness,
                    fill=color)
            elif len(bbox) == 8:
                x1, y1, x2, y2, x3, y3, x4, y4 = bbox
                draw.line(
                    [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                    width=2,
                    fill=color)
                xmin = min(x1, x2, x3, x4)
                ymin = min(y1, y2, y3, y4)

            # draw label
            text = "{} {:.4f}".format(label, score)
            tw, th = self.imagedraw_textsize_c(draw, text)
            draw.rectangle(
                [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
            draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))
        return np.array(im)

