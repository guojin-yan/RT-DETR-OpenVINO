// cpp.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "process.h"
#include "rtdert_predictor.h"


void RT_DETR() {
    INFO("This is an RT-DETR model deployment case using C++!");
    //std::string model_path = "E:\\Model\\rtdetr_r50vd_6x_coco.onnx";
    std::string model_path = "E:\\Model\\RT-DETR\\rtdetr_r50vd_6x_coco.xml";
    std::string image_path = "E:\\GitSpace\\RT-DETR-OpenVINO\\image\\000000570688.jpg";
    std::string label_path = "E:\\GitSpace\\RT-DETR-OpenVINO\\image\\COCO_lable.txt";
    RTDETRPredictor predictor(model_path, label_path, "CPU", false);
    //RTDETRPredictor predictor(model_path, label_path, "CPU", true);
    cv::Mat image = cv::imread(image_path);
    cv::Mat result_mat = predictor.predict(image);
    cv::imshow("C++ deploy RT-DETR result", result_mat);
    cv::waitKey(0);
}

int main()
{
    RT_DETR();
    getchar();
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
