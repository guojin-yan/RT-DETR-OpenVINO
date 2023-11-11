// cpp.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "process.h"
#include "rtdert_predictor.h"


void RT_DETR(std::string model_path, std::string image_path, std::string label_path,bool post_flag) {
    INFO("This is an RT-DETR model deployment case using C++!");

    //std::string image_path = "E:\\GitSpace\\RT-DETR-OpenVINO\\image\\000000570688.jpg";
    //std::string label_path = "E:\\GitSpace\\RT-DETR-OpenVINO\\image\\COCO_lable.txt";
    cv::Mat image = cv::imread(image_path);
    cv::Mat result_mat;
    if (post_flag) {
        std::cout << post_flag << std::endl;
        //std::string model_path = "E:\\Model\\rtdetr_r50vd_6x_coco.onnx";
        RTDETRPredictor predictor(model_path, label_path, "GPU.0", true);
        result_mat = predictor.predict(image);
    }
    else {
        //std::string model_path = "E:\\Model\\RT-DETR\\rtdetr_r50vd_6x_coco.xml";
        RTDETRPredictor predictor(model_path, label_path, "CPU", false);
        result_mat = predictor.predict(image);
    }
    cv::imshow("C++ deploy RT-DETR result", result_mat);
    cv::waitKey(0);
}

int main(int argc, char* argv[])
{
    if (argc < 5) {
        INFO("Please enter the correct parameters.");
        INFO("For example:");
        INFO("  rt-detr_openvino_cpp.exe [model path] [image path] [lable path] [post flag(1/0)].");
        return 0;
    }
    bool b;
    // 錯誤輸入返回 false
    std::istringstream(argv[4]) >> b;
    RT_DETR(argv[1], argv[2], argv[3], b);
    getchar();
}

