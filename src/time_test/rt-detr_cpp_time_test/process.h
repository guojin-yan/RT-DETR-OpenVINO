// Copyright(©) 2023, Company All Rights Reserved 
// -*- coding: utf-8 -*-
// @Time    : 2023/10/15  21:45:49
// @Brief  : This is class file.
// @File    : process.h
// @Version : 1.0
// @Author  : Yan Guojin
// @E-mail	: guojin_yjs@cumt.edu.cn
// @GitHub	: https://github.com/guojin-yan
// @Description : 


#ifndef __PROCESS_H__
#define __PROCESS_H__

#include <algorithm>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"

#define INFO(...) \
        std::cout << "[INFO]  " << __VA_ARGS__ << std::endl;


struct ResultData {
    std::vector<int> clsids;
    std::vector<std::string> labels;
    std::vector<cv::Rect> bboxs;
    std::vector<float> scores;
    ResultData() {}
};


class RTDETRProcess
{
public:
    RTDETRProcess() {}
    RTDETRProcess(cv::Size target_size, std::string label_path = NULL, float threshold = 0.5,
        cv::InterpolationFlags interpf = cv::INTER_LINEAR);
    cv::Mat preprocess(cv::Mat image);
    ResultData postprocess(float* score, float* bboxs, bool post_flag);
    std::vector<float> get_im_shape() { return im_shape; }
    std::vector<float> get_input_shape() { return { (float)target_size.width ,(float)target_size.height }; }
    std::vector<float> get_scale_factor() { return scale_factor; }
    cv::Mat draw_box(cv::Mat image, ResultData results);

private:
    void read_labels(std::string label_path);
    template<class T>
    float sigmoid(T data) {
        return 1.0f / (1 + std::exp(-data));
    }
    template<class T>
    int argmax(T* data, int length) {
        std::vector<T> arr(data, data + length);
        return (int)(std::max_element(arr.begin(), arr.end()) - arr.begin());
    }

private:
    cv::Size target_size;               // The model input size.
    std::vector<std::string> labels;    // The model classification label.
    float threshold;                    // The threshold parameter.
    cv::InterpolationFlags interpf;     // The image scaling method.
    std::vector<float> im_shape;
    std::vector<float> scale_factor;
};


#endif // !__PROCESS_H__




