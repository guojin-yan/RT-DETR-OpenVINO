// Copyright(©) 2023, Company All Rights Reserved 
// -*- coding: utf-8 -*-
// @Time    : 2023/10/15  21:44:38
// @Brief  : This is common class.
// @File    : process.cpp
// @Version : 1.0
// @Author  : Yan Guojin
// @E-mail	: guojin_yjs@cumt.edu.cn
// @GitHub	: https://github.com/guojin-yan
// @Description : 

#include "process.h"
#include <fstream>
#include <iostream>
#include "opencv2/opencv.hpp"


/**
 * The RTDETRProcess constructor initializes the target size, threshold, and interpolation flags, and
 * reads labels from a given file path if provided.
 * 
 * @param target_size The target size is the desired size of the processed image. It is specified as a
 * cv::Size object, which contains the width and height of the target size.
 * @param label_path The `label_path` parameter is a string that represents the file path to the labels
 * file. This file contains the labels or classes that the model can predict.
 * @param threshold The threshold is a value used to determine the minimum confidence level required
 * for an object detection to be considered valid. Any detection with a confidence level below the
 * threshold will be ignored.
 * @param interpf The `interpf` parameter is of type `cv::InterpolationFlags` and is used to specify
 * the interpolation method to be used when resizing the input image. It determines how the pixels are
 * interpolated when the image is resized to the `target_size`.
 */
RTDETRProcess::RTDETRProcess(cv::Size target_size, std::string label_path, 
	float threshold, cv::InterpolationFlags interpf)
	: target_size(target_size), threshold(threshold), interpf(interpf){
    if (!label_path.empty()) {
        read_labels(label_path);
    }
}

/**
 * The function preprocesses an input image by resizing it, converting it to RGB color space, and
 * normalizing its pixel values.
 * 
 * @param image The input image that needs to be preprocessed.
 * 
 * @return a cv::Mat object, which is the preprocessed image.
 */
cv::Mat RTDETRProcess::preprocess(cv::Mat image){
    im_shape = { (float)image.rows, (float)image.cols };
    scale_factor = { 640.0f / (float)image.rows, 640.0f / (float)image.cols};
    cv::Mat blob_image;
    cv::cvtColor(image, blob_image, cv::COLOR_BGR2RGB); 
    cv::resize(blob_image, blob_image, target_size, 0, 0, cv::INTER_LINEAR);
    std::vector<cv::Mat> rgb_channels(3);
    cv::split(blob_image, rgb_channels);
    for (auto i = 0; i < rgb_channels.size(); i++) {
        rgb_channels[i].convertTo(rgb_channels[i], CV_32FC1, 1.0 / 255.0);
    }
    cv::merge(rgb_channels, blob_image);
    return blob_image;
}

/**
 * The function `postprocess` takes in an array of scores and bounding box coordinates, and returns a
 * `ResultData` object containing the filtered results based on a threshold and post_flag.
 * 
 * @param score A pointer to an array of floats representing the scores of the detected objects.
 * @param bbox The `bbox` parameter is an array of floats that represents the bounding box coordinates
 * for each detected object. Each object is represented by 4 values in the array, which are the
 * x-coordinate, y-coordinate, width, and height of the bounding box. The `bbox` array has a length of
 * @param post_flag The "post_flag" parameter is a boolean flag that determines whether to use
 * post-processing or not. If it is set to true, the function will perform post-processing on the input
 * data. If it is set to false, the function will not perform post-processing and will use a different
 * method to process
 * 
 * @return an object of type ResultData.
 */
ResultData RTDETRProcess::postprocess(float* score, float* bbox, bool post_flag)
{
    ResultData result;
    if (post_flag) {
        for (int i = 0; i < 300; ++i) {
            if (score[6 * i + 1] > threshold) {
                result.clsids.push_back((int)score[6 * i ]);
                result.labels.push_back(labels[(int)score[6 * i]]);
                result.bboxs.push_back(cv::Rect(score[6 * i + 2], score[6 * i + 3],
                    score[6 * i + 4] - score[6 * i + 2],
                    score[6 * i + 5] - score[6 * i + 3]));
                result.scores.push_back(score[6 * i + 1]);
            }
        }
    } else {
        for (int i = 0; i < 300; ++i) {
            float s[80];
            for (int j = 0; j < 80; ++j) {
                s[j] = score[80 * i + j];
            }
            int clsid = argmax<float>(s, 80);
            float max_score = sigmoid<float>(s[clsid]);
            if (max_score > threshold) {
                result.clsids.push_back(clsid);
                result.labels.push_back(labels[clsid]);
                float cx = bbox[4 * i] * 640.0 / scale_factor[1];
                float cy = bbox[4 * i + 1] * 640.0 / scale_factor[0];
                float w = bbox[4 * i + 2] * 640.0 / scale_factor[1];
                float h = bbox[4 * i + 3] * 640.0 / scale_factor[0];
                result.bboxs.push_back(cv::Rect((int)(cx - w / 2), (int)(cy - h / 2), w, h));
                result.scores.push_back(max_score);
            }
        }
    }
    return result;
}


/**
 * The function `draw_box` takes an input image and a set of results, and draws bounding boxes with
 * labels and scores on the image.
 * 
 * @param image The input image on which the bounding boxes will be drawn.
 * @param results The "results" parameter is an object of type "ResultData" which contains the
 * following attributes:
 * 
 * @return a cv::Mat object, which is a matrix representing an image.
 */
cv::Mat RTDETRProcess::draw_box(cv::Mat image, ResultData results) {
    cv::Mat re_image = image.clone();
    INFO("Infer result:")
    for (int i = 0; i < results.clsids.size(); ++i) {
        int clsid = results.clsids[i];
        std::string label = results.labels[i];
        cv::Rect bbox = results.bboxs[i];
        float score = results.scores[i];
        
        std::string score_str = std::to_string(score);
        std::string text = label;
        text += ("  " + score_str.substr(0, score_str.find(".") + 4));
        cv::rectangle(re_image, bbox, cv::Scalar(255, 0, 0), 1);
        int y = 5;
        cv::Size text_size = cv::getTextSize(text, 0, 0.4, 1, &y);
        cv::Rect rec(bbox.tl().x, bbox.tl().y - text_size.height, text_size.width, text_size.height);
        std::vector<cv::Point>  contour;
        contour.push_back(rec.tl());
        contour.push_back(cv::Point(rec.tl().x + rec.width, rec.tl().y));
        contour.push_back(cv::Point(rec.tl().x + rec.width, rec.tl().y + rec.height));
        contour.push_back(cv::Point(rec.tl().x, rec.tl().y + rec.height));

        cv::fillConvexPoly(re_image, contour, cv::Scalar(0, 0, 0));
        cv::putText(re_image, text, cv::Point(bbox.tl().x, bbox.tl().y),
            1, 0.7, cv::Scalar(255, 255, 255), 1);
        
        std::string msg = "  class_id : " + std::to_string(clsid) + ", label : " + label + 
            ", confidence : " + score_str.substr(0, score_str.find(".") + 4) + ", left_top : [" +
            std::to_string(bbox.tl().x) + ", " + std::to_string(bbox.tl().y) +"], right_bottom: [" + 
            std::to_string(bbox.br().x) + ", " + std::to_string(bbox.br().y) + "]";

       /* INFO(msg);*/
    }
    return re_image;
}

/**
 * The function reads labels from a file and stores them in a vector.
 * 
 * @param label_path The parameter `label_path` is a string that represents the path to the file
 * containing the labels.
 */
void RTDETRProcess::read_labels(std::string label_path){
    std::ifstream file(label_path);
    if (file){
        std::string str;
        while (std::getline(file, str)) {
            labels.push_back(str);
        }
    }
    else {
        std::cout << "Open file faild." << std::endl;
    }
    file.close();
}




