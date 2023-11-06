// Copyright(©) 2023, Company All Rights Reserved 
// -*- coding: utf-8 -*-
// @Time    : 2023/10/15  21:50:13
// @Brief  : This is common class.
// @File    : rtdert_predictor.cpp
// @Version : 1.0
// @Author  : Yan Guojin
// @E-mail	: guojin_yjs@cumt.edu.cn
// @GitHub	: https://github.com/guojin-yan
// @Description : 

#include "rtdert_predictor.h"
#include <opencv2/opencv.hpp>
#include "process.h"
#include <windows.h>

/**
 * The RTDETRPredictor constructor initializes the RTDETRPredictor object with the specified model
 * path, label path, device name, and post_flag.
 * 
 * @param model_path The path to the model file that will be used for prediction.
 * @param label_path The `label_path` parameter is the path to the file that contains the labels or
 * classes for the objects that the model can detect. This file typically contains a list of class
 * names, with each name on a separate line.
 * @param device_name The device name is the name of the device on which the model will be run. It can
 * be "CPU", "GPU.0", etc., depending on the available hardware and the OpenVINO installation.
 * @param post_flag The `post_flag` parameter is a Boolean flag indicating whether the inference model 
 * includes a network layer for post-processing the inference results. If `post_flag` is set to 'True', 
 * the inference model includes a network layer for post-processing the inference results. If 'post_flag'
 * is set to False, the inference model does not include a network layer for post-processing the inference 
 * results. Default value is True.
 */
RTDETRPredictor::RTDETRPredictor(std::string model_path, std::string label_path, 
std::string device_name, bool post_flag)
	:post_flag(post_flag){
    INFO("Model path: " + model_path);
    INFO("Device name: " + device_name);
	// The `read_model` function reads the model file and returns a shared pointer to an
    // instance of the `ov::Model` class, which represents the model. 
    clock_t t1, t2;
    t1 = clock();
    model = core.read_model(model_path);
	// The line is compiling the model for a specific device. 
    compiled_model = core.compile_model(model, device_name);
	// Creates an inference request object for the compiled model. This request object is
    // used to perform inference on the model by providing input data and retrieving the output data.
    infer_request = compiled_model.create_infer_request();
    t2 = clock();
    load_model += (t2 - t1);
    // Creating an instance of the `RTDETRProcess` class and assigning it to the `rtdetr_process` variable.
    rtdetr_process = RTDETRProcess(cv::Size(640, 640), label_path, 0.5);
}

/**
 * The `predict` function takes an input image, preprocesses it, performs inference using a pre-trained
 * model, postprocesses the output, and returns the image with bounding boxes drawn around detected
 * objects.
 * 
 * @param image The input image that needs to be processed and predicted by the RTDETR model.
 * 
 * @return a cv::Mat object, which represents an image.
 */
cv::Mat RTDETRPredictor::predict(cv::Mat image){
    clock_t t1, t2;
    t1 = clock();
    cv::Mat blob_image = rtdetr_process.preprocess(image);
    t2 = clock();
    process_image += (t2 - t1);
    t1 = clock();
    if (post_flag) {
        ov::Tensor image_tensor = infer_request.get_tensor("image");
        ov::Tensor shape_tensor = infer_request.get_tensor("im_shape");
        ov::Tensor scale_tensor = infer_request.get_tensor("scale_factor");
        image_tensor.set_shape({ 1,3,640,640 });
        shape_tensor.set_shape({ 1,2 });
        scale_tensor.set_shape({ 1,2 });
        fill_tensor_data_image(image_tensor, blob_image);
        fill_tensor_data_float(shape_tensor, rtdetr_process.get_input_shape().data(), 2);
        fill_tensor_data_float(scale_tensor, rtdetr_process.get_scale_factor().data(), 2);
    } else {
        ov::Tensor image_tensor = infer_request.get_input_tensor();
        image_tensor.set_shape({ 1,3,640,640 });
        fill_tensor_data_image(image_tensor, blob_image);
    }
    t2 = clock();
    load_data += (t2 - t1);
    t1 = clock();
    infer_request.infer();
    t2 = clock();
    infer += (t2 - t1);

    ResultData results;

    t1 = clock();
    if (post_flag) {
        ov::Tensor output_tensor = infer_request.get_tensor("reshape2_69.tmp_0");
        float result[6 * 300] = {0};
        for (int i = 0; i < 6 * 300; ++i) {
            result[i] = output_tensor.data<float>()[i];
        }
        results = rtdetr_process.postprocess(result, nullptr, true);
    } else {
        ov::Tensor score_tensor = infer_request.get_tensor(model->outputs()[1].get_any_name());
        ov::Tensor bbox_tensor = infer_request.get_tensor(model->outputs()[0].get_any_name());
        float score[300 * 80] = {0};
        float bbox[300 * 4] = {0};
        for (int i = 0; i < 300; ++i) {
            for (int j = 0; j < 80; ++j) {
                score[80 * i + j] = score_tensor.data<float>()[80 * i + j];
            }
            for (int j = 0; j < 4; ++j) {
                bbox[4 * i + j] = bbox_tensor.data<float>()[4 * i + j];
            }
        }
        results = rtdetr_process.postprocess(score, bbox, false);
    }
    t2 = clock();
    post_process += (t2 - t1);
    return rtdetr_process.draw_box(image, results);
}

/**
 * The function `pritf_model_info` prints information about an inference model, including its name,
 * input details (name, type, shape), and output details (name, type, shape).
 * 
 * @param model A shared pointer to an instance of the `ov::Model` class.
 */
void RTDETRPredictor::pritf_model_info(std::shared_ptr<ov::Model> model) {
    INFO("Inference Model");
    INFO("  Model name: " + model->get_friendly_name());
    INFO("  Input:");
    std::vector<ov::Output<ov::Node>> inputs = model->inputs();
    for (auto input : inputs) {
        INFO("     name: " + input.get_any_name());
        INFO("     type: " + input.get_element_type().c_type_string());
        INFO("     shape: " + input.get_partial_shape().to_string());
    }
    INFO("  Output:");
    std::vector<ov::Output<ov::Node>> outputs = model->outputs();
    for (auto output : outputs) {
        INFO("     name: " + output.get_any_name());
        INFO("     type: " + output.get_element_type().c_type_string());
        INFO("     shape: " + output.get_partial_shape().to_string());
    }
}

/**
 * The function `fill_tensor_data_image` fills an OpenVINO tensor with data from an input image.
 * 
 * @param input_tensor The input tensor is an instance of the ov::Tensor class, which represents a
 * multi-dimensional array of data. It is used to store the input data for the model.
 * @param input_image The input_image parameter is a cv::Mat object, which represents an image in
 * OpenCV. It is a multi-channel image with floating-point pixel values.
 */
void RTDETRPredictor::fill_tensor_data_image(ov::Tensor& input_tensor, 
    const cv::Mat& input_image) {
    
    // Retrieving the shape of the input tensor. The `get_shape()` function returns an `ov::Shape` 
    // object, which represents the shape of the tensor. The shape is a vector of integers that 
    // specifies the size of each dimension of the tensor. In this case, `tensor_shape` is assigned 
    // the shape of the input tensor, which is then used to determine the width, height, and number 
    // of channels of the input image. 
    ov::Shape tensor_shape = input_tensor.get_shape();
    const size_t width = tensor_shape[3]; 
    const size_t height = tensor_shape[2];
    const size_t channels = tensor_shape[1]; 
    // Retrieving a pointer to the data buffer of the input tensor. It is used to access and modify 
    // the values of the tensor. By assigning the pointer to `input_tensor_data`, you can directly 
    // manipulate the data in the tensor using array indexing. 
    float* input_tensor_data = input_tensor.data<float>();
    // The code snippet is filling the input tensor with data from the input image. 
    for (size_t c = 0; c < channels; c++) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                input_tensor_data[c * width * height + h * width + w] = input_image.at<cv::Vec<float, 3>>(h, w)[c];
            }
        }
    }
}
/**
 * The function fills a tensor with float data from an input array.
 * 
 * @param input_tensor The input tensor to be filled with data.
 * @param input_data A pointer to an array of float values representing the input data.
 * @param data_size The parameter "data_size" represents the size of the input data array. It indicates
 * the number of elements in the array that need to be copied to the input tensor.
 */
void RTDETRPredictor::fill_tensor_data_float(ov::Tensor& input_tensor, float* input_data, int data_size) {
    // Retrieving a pointer to the data buffer of the input tensor. 
    float* input_tensor_data = input_tensor.data<float>();
    // Filling a tensor with float data from an input array.
    for (int i = 0; i < data_size; i++) {
        input_tensor_data[i] = input_data[i];
    }
}

double RTDETRPredictor::load_model = 0.0;
double RTDETRPredictor::process_image = 0.0;
double RTDETRPredictor::load_data = 0.0;
double RTDETRPredictor::infer = 0.0;
double RTDETRPredictor::post_process = 0.0;