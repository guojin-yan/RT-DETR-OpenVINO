// Copyright(©) 2023, Company All Rights Reserved 
// -*- coding: utf-8 -*-
// @Time    : 2023/10/15  21:50:05
// @Brief  : This is common class.
// @File    : rtdert_predictor.h
// @Version : 1.0
// @Author  : Yan Guojin
// @E-mail	: guojin_yjs@cumt.edu.cn
// @GitHub	: https://github.com/guojin-yan
// @Description : 
#ifndef __RTDETRPREDICTOR_H__
#define __RTDETRPREDICTOR_H__
#include "openvino/openvino.hpp"
#include "opencv2/opencv.hpp"
#include "process.h"
class RTDETRPredictor
{
public:
    RTDETRPredictor(std::string model_path, std::string label_path, 
        std::string device_name = "CPU", bool postprcoess = true);

    cv::Mat predict(cv::Mat image);
private:
    void pritf_model_info(std::shared_ptr<ov::Model> model);

    void fill_tensor_data_image(ov::Tensor& input_tensor, const cv::Mat& input_image);

    void fill_tensor_data_float(ov::Tensor& input_tensor, float* input_data, int data_size);

private:
    RTDETRProcess rtdetr_process;
    bool post_flag;
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
public:
    static double load_model;
    static double process_image;
    static double load_data;
    static double infer;
    static double post_process;
};

#endif // __RTDETRPREDICTOR__



