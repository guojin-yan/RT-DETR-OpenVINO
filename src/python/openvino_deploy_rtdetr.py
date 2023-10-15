# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/14  14:43:04
# @Author  : Yan Guojin
# @File    : openvino_deploy_rtdetr.py
# @E-mail	: guojin_yjs@cumt.edu.cn
# @GitHub	: https://github.com/guojin-yan
# @Description : 

import cv2 as cv
import numpy as np
from openvino.runtime import Core
from process import RtdetrProcess, print_info


def rtdert_infer(model_path, image_path, device_name, lable_path, postprocess=True):
    """
    The `rtdert_infer` function performs inference using a pre-trained model on an input image,
    preprocesses the input data, runs the inference, and then post-processes the results if specified.
    
    Args:
      model_path: The path to the model file.
      image_path: The path to the input image that you want to perform inference on.
      device_name: The `device_name` parameter specifies the device on which the model will be executed.
    It can be set to "CPU", "GPU.0".
      lable_path: The `lable_path` parameter is the path to the file that contains the labels or classes
    for the objects that the model can detect.
      postprocess: The `postprocess` parameter is a Boolean flag indicating whether the inference model 
    includes a network layer for post-processing the inference results. If `postprocess` is set to 'True', 
    the inference model includes a network layer for post-processing the inference results. If 'postprocess'
    is set to False, the inference model does not include a network layer for post-processing the inference 
    results. Default value is True.
    """
    '''-------------------1. Initialize and Read Model ----------------------'''
    print_info("Model path: " + model_path)
    print_info("Device name: " + device_name)
    ie_core = Core()
    model = ie_core.read_model(model=model_path)
    compiled_model = ie_core.compile_model(model=model, device_name=device_name)

    '''-------------------2. Preprocessing model input data ----------------------'''
    print_info("The input path: " + image_path)
    image = cv.imread(image_path)
    rtdetr_process = RtdetrProcess([640,640],lable_path)
    im, im_info= rtdetr_process.preprocess(image)
    inputs = dict()
    inputs["image"] = np.array(im).astype('float32')
    
    '''-------------------3. Infer ----------------------'''
    if(postprocess):
        inputs["scale_factor"] = np.array(im_info['scale_factor']).reshape(1,2).astype('float32')
        inputs["im_shape"] = np.array([640.0,640.0]).reshape(1,2).astype('float32')
        results = compiled_model(inputs=inputs)
    else:
        results = compiled_model(inputs=inputs)
    '''-------------------5. Post processing prediction results ----------------------'''
    if(postprocess):
        re = rtdetr_process.postprocess(results[compiled_model.output(0)])
        new_image=rtdetr_process.draw_box(image,re)
        cv.imshow("result",new_image)
        cv.waitKey(0)
    else:
        re=rtdetr_process.postprocess(results[compiled_model.output(1)][0],results[compiled_model.output(0)][0])
        new_image=rtdetr_process.draw_box(image,re)
        cv.imshow("Python deploy RT-DETR result",new_image)
        cv.waitKey(0)



    