#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "process.h"
#include "rtdert_predictor.h"


void RT_DETR(std::string model_path, std::string image_path, std::string label_path, bool post_flag) {
    INFO("This is an RT-DETR model deployment case using C++!");

    //std::string image_path = "E:\\GitSpace\\RT-DETR-OpenVINO\\image\\000000570688.jpg";
    //std::string label_path = "E:\\GitSpace\\RT-DETR-OpenVINO\\image\\COCO_lable.txt";
    cv::Mat image = cv::imread(image_path);
    cv::Mat result_mat;
    int n = 100;

    for (int i = 0; i < n; ++i) {
        std::cout << "Model predict: " << i << std::endl;
        //std::string model_path = "E:\\Model\\rtdetr_r50vd_6x_coco.onnx";
        RTDETRPredictor predictor(model_path, label_path, "CPU", true);
        result_mat = predictor.predict(image);
    }
    std::cout << RTDETRPredictor::load_model/n << std::endl;
    std::cout << RTDETRPredictor::process_image / n << std::endl;
    std::cout << RTDETRPredictor::load_data / n << std::endl;
    std::cout << RTDETRPredictor::infer / n << std::endl;
    std::cout << RTDETRPredictor::post_process / n << std::endl;

}

int main(int argc, char* argv[])
{
    RT_DETR("E:\\Model\\RT-DETR\\RTDETR\\rtdetr_r34vd_6x_coco.xml",
        "E:\\GitSpace\\RT-DETR-OpenVINO\\image\\000000570688.jpg",
        "E:\\GitSpace\\RT-DETR-OpenVINO\\image\\COCO_lable.txt", true);
}


