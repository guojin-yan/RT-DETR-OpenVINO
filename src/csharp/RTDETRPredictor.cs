// Copyright(©) 2023, Company All Rights Reserved 
// -*- coding: utf-8 -*-
// @Time    : 2023/10/17  14:13:44
// @Brief  : This is common class.
// @File    : RTDETRPredictor.cs
// @Version : 1.0
// @Author  : Yan Guojin
// @E-mail	: guojin_yjs@cumt.edu.cn
// @GitHub	: https://github.com/guojin-yan
// @Description : 
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using OpenVinoSharp;
using static rt_detr_openvino_csharp.Msg;

namespace rt_detr_openvino_csharp
{
    public class RTDETRPredictor
    {

        /// <summary>
        /// The RTDETRPredictor function initializes the RTDETR model, compiles it for a specific device, and
        /// creates an inference request object for performing inference on the model.
        /// </summary>
        /// <param name="model_path">The path to the model file. This file contains the trained model that will
        /// be used for prediction.</param>
        /// <param name="label_path">The `label_path` parameter is a string that represents the path to the file
        /// containing the labels or classes for the model. This file typically contains a list of class names
        /// that the model can predict.</param>
        /// <param name="device_name">The `device_name` parameter is used to specify the device on which the
        /// model will be executed. It has a default value of "CPU", which means the model will be executed on
        /// the CPU. Other possible values could be "GPU" or specific device names like "GPU:0" or "</param>
        /// <param name="postprcoess">The `postprcoess` parameter is a boolean flag that determines whether
        /// post-processing should be applied to the output of the model. If `postprcoess` is set to `true`,
        /// post-processing will be applied. If it is set to `false`, no post-processing will be applied</param>
        public RTDETRPredictor(string model_path, string label_path,
         string device_name = "CPU", bool postprcoess = true)
        {
            INFO("Model path: " + model_path);
            INFO("Device name: " + device_name);
            core = new Core();
            // The `read_model` function reads the model file and returns a shared pointer to an
            // instance of the `ov::Model` class, which represents the model. 
            model = core.read_model(model_path);
            pritf_model_info(model);
            // The line is compiling the model for a specific device. 
            compiled_model = core.compile_model(model, device_name);
            // Creates an inference request object for the compiled model. This request object is
            // used to perform inference on the model by providing input data and retrieving the output data.
            infer_request = compiled_model.create_infer_request();
            // Creating an instance of the `RTDETRProcess` class and assigning it to the `rtdetr_process` variable.
            rtdetr_process = new RTDETRProcess(new Size(640, 640), label_path, 0.5f);
            this.post_flag = postprcoess;
        }

       /// <summary>
       /// The function takes an input image, preprocesses it, performs inference using a pre-trained
       /// model, and returns the image with bounding boxes drawn around detected objects.
       /// </summary>
       /// <param name="Mat">The `Mat` class is a data structure in OpenCV that represents an image
       /// matrix. It is used to store and manipulate image data.</param>
       /// <returns>
       /// The method is returning a `Mat` object, which is an image with bounding boxes drawn on it.
       /// </returns>
        public Mat predict(Mat image)
        {
            Mat blob_image = rtdetr_process.preprocess(image.Clone());
            if (post_flag)
            {
                Tensor image_tensor = infer_request.get_tensor("image");
                Tensor shape_tensor = infer_request.get_tensor("im_shape");
                Tensor scale_tensor = infer_request.get_tensor("scale_factor");
                image_tensor.set_shape(new Shape(new List<long>{ 1, 3, 640, 640 }));
                shape_tensor.set_shape(new Shape(new List<long> { 1,2 }));
                scale_tensor.set_shape(new Shape(new List<long> { 1,2 }));
                fill_tensor_data_image(image_tensor, blob_image);
                fill_tensor_data_float(shape_tensor, rtdetr_process.get_input_shape().ToArray(), 2);
                fill_tensor_data_float(scale_tensor, rtdetr_process.get_scale_factor().ToArray(), 2);
            }
            else
            {
                Tensor image_tensor = infer_request.get_input_tensor();
                image_tensor.set_shape(new Shape(new List<long> { 1, 3, 640, 640 }));
                fill_tensor_data_image(image_tensor, blob_image);
            }
            infer_request.infer();
            ResultData results;
            if (post_flag)
            {
                Tensor output_tensor = infer_request.get_output_tensor(0);
                float[] result = output_tensor.get_data<float>(300 * 6);

                results = rtdetr_process.postprocess(result, null, true);
            }
            else
            {
                Tensor score_tensor = infer_request.get_tensor(model.outputs()[1].get_any_name());
                Tensor bbox_tensor = infer_request.get_tensor(model.outputs()[0].get_any_name());
                float[] score = score_tensor.get_data<float>(300 * 80);
                float[] bbox = bbox_tensor.get_data<float>(300 * 4);
                results = rtdetr_process.postprocess(score, bbox, false);
            }
            return rtdetr_process.draw_box(image, results);
        }

       /// <summary>
       /// The function `pritf_model_info` prints information about an inference model, including its
       /// name, input details, and output details.
       /// </summary>
       /// <param name="Model">The `Model` parameter represents an inference model. It contains
       /// information about the model's inputs and outputs.</param>
        private void pritf_model_info(Model model)
        {
            INFO("Inference Model");
            INFO("  Model name: " + model.get_friendly_name());
            INFO("  Input:");
            List<Input> inputs = model.inputs();
            foreach (var input in inputs)
            {
                INFO("     name: " + input.get_any_name());
                INFO("     type: " + input.get_element_type().c_type_string());
                INFO("     shape: " + input.get_partial_shape().to_string());
            }
            INFO("  Output:");
            List<Output> outputs = model.outputs();
            foreach (var output in outputs)
            {
                INFO("     name: " + output.get_any_name());
                INFO("     type: " + output.get_element_type().c_type_string());
                INFO("     shape: " + output.get_partial_shape().to_string());
            }
        }

        /// <summary>
        /// The function fills a tensor with data from an input image.
        /// </summary>
        /// <param name="Tensor">The `Tensor` parameter is an object that represents a multi-dimensional array
        /// of data. It is used to store and manipulate numerical data, such as images or other types of data
        /// used in machine learning or computer vision tasks.</param>
        /// <param name="Mat">The "Mat" parameter is an object representing an image in OpenCV. It is a
        /// matrix-like structure that stores the pixel values of the image.</param>
        private void fill_tensor_data_image(Tensor input_tensor, Mat input_image)
        {
            float[] data = new float[input_tensor.get_size()];
            Marshal.Copy(input_image.Ptr(0), data, 0, (int)input_tensor.get_size());
            input_tensor.set_data(data);
        }

        /// <summary>
        /// The function "fill_tensor_data_float" fills a given Tensor object with float data.
        /// </summary>
        /// <param name="Tensor">The `Tensor` parameter is an object that represents a multi-dimensional array
        /// or matrix. It is used to store and manipulate numerical data in machine learning and deep learning
        /// frameworks.</param>
        /// <param name="input_data">An array of float values that you want to set as the data for the
        /// input_tensor.</param>
        /// <param name="data_size">The parameter "data_size" represents the size of the input data array. It
        /// indicates the number of elements in the input_data array that should be used to fill the
        /// input_tensor.</param>
        private void fill_tensor_data_float(Tensor input_tensor, float[] input_data, int data_size)
        {
            input_tensor.set_data(input_data);
        }


        RTDETRProcess rtdetr_process;
        bool post_flag;
        Core core;
        Model model;
        CompiledModel compiled_model;
        InferRequest infer_request;
    }
}
