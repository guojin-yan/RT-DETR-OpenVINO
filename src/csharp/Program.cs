// Copyright(©) 2023, Company All Rights Reserved 
// -*- coding: utf-8 -*-
// @Time    : 2023/10/17  14:13:25
// @Brief  : This is common class.
// @File    : Program.cs
// @Version : 1.0
// @Author  : Yan Guojin
// @E-mail	: guojin_yjs@cumt.edu.cn
// @GitHub	: https://github.com/guojin-yan
// @Description : 
using OpenCvSharp;
using static rt_detr_openvino_csharp.Msg;
namespace rt_detr_openvino_csharp
{
    internal class Program
    {
        static void RT_DETR(string model_path, string image_path, string label_path, bool post_flag)
        {
            INFO("Hello, World!");
            //string image_path = "E:\\GitSpace\\RT-DETR-OpenVINO\\image\\000000570688.jpg";
            //string label_path = "E:\\GitSpace\\RT-DETR-OpenVINO\\image\\COCO_lable.txt";
            Mat image = Cv2.ImRead(image_path);
            Mat result_mat = new Mat();
            if (post_flag)
            {
                //  string model_path  = "E:\\Model\\rtdetr_r50vd_6x_coco.onnx";
                RTDETRPredictor predictor = new RTDETRPredictor(model_path, label_path, "CPU", true);
                result_mat = predictor.predict(image);
            }
            else
            {
                // string model_path = "E:\\Model\\RT-DETR\\rtdetr_r50vd_6x_coco.xml";
                RTDETRPredictor predictor = new RTDETRPredictor(model_path, label_path, "CPU", false);
                result_mat = predictor.predict(image);
            }
            Cv2.ImShow("C# deploy RT-DETR result", result_mat);
           Cv2.WaitKey(0);
        }
        static void Main(string[] args)
        {
            if (args.Length < 4) {
                INFO("Please enter the correct parameters.");
                INFO("For example:");
                INFO("  dotnet run [model path] [image path] [lable path] [post flag(1/0)].");
            }
            RT_DETR(args[0], args[1], args[2], Convert.ToBoolean(Convert.ToInt32(args[3])));
        }
    }
}