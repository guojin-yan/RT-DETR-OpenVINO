# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/14  14:42:16
# @Author  : Yan Guojin
# @File    : main.py
# @E-mail	: guojin_yjs@cumt.edu.cn
# @GitHub	: https://github.com/guojin-yan
# @Description : 
import sys
from process import print_info
from openvino_deploy_rtdetr import rtdert_infer

def main(model_path, image_path, lable_path, postprocess = True):
    """
    The main function prints a greeting message, sets the paths for a model, an image, and a label file,
    and then calls the rtdert_infer function with these paths and some additional parameters.
    """
    print_info("This is an RT-DETR model deployment case using Python!")
    if(postprocess):
        # model_path = "E:\\Model\\rtdetr_r50vd_6x_coco.onnx"
        # image_path = "E:\\GitSpace\\RT-DETR-OpenVINO\\image\\000000570688.jpg"
        # lable_path = "E:\\GitSpace\\OpenVINO-CSharp-API\dataset\\lable\\COCO_lable.txt"
        rtdert_infer(model_path, image_path, "CPU", lable_path,True)
    else:
        # model_path = "E:\\Model\\RT-DETR\\rtdetr_r50vd_6x_coco.xml"
        # image_path = "E:\\GitSpace\\RT-DETR-OpenVINO\\image\\000000570688.jpg"
        # lable_path = "E:\\GitSpace\\OpenVINO-CSharp-API\dataset\\lable\\COCO_lable.txt"
        rtdert_infer(model_path, image_path, "CPU", lable_path,False)


# The `if __name__ == '__main__':` statement is used to check whether the current script is being run
# directly or being imported as a module.
if __name__ == '__main__':
    if (len(sys.argv) < 5) :
        print_info("Please enter the correct parameters.")
        print_info("For example:")
        print_info("  python main.py [model path] [image path] [lable path] [post flag(1/0)].")
    else:
        b=False
        if sys.argv[4] == "0":
            b=False
        elif sys.argv[4] == "1":
            b=True
        
        main(sys.argv[1],sys.argv[2],sys.argv[3],b)