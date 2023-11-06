// Copyright(©) 2023, Company All Rights Reserved 
// -*- coding: utf-8 -*-
// @Time    : 2023/10/17  14:16:52
// @Brief  : This is common class.
// @File    : RTDETRProcess.cs
// @Version : 1.0
// @Author  : Yan Guojin
// @E-mail	: guojin_yjs@cumt.edu.cn
// @GitHub	: https://github.com/guojin-yan
// @Description : 
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Reflection.Emit;
using System.Reflection.Metadata.Ecma335;
using System.Security.Claims;
using System.Text;
using System.Threading.Tasks;
using static OpenVinoSharp.Ov;
using static rt_detr_time_text.Msg;

namespace rt_detr_time_text
{
    public static class Msg {
        public static void INFO(string msg)
        {
            Console.WriteLine("[INFO]  " + msg);
        }
    }
    public struct ResultData
    {
        public List<int> clsids;
        public List<string> labels;
        public List<Rect> bboxs;
        public List<float> scores;
        public ResultData() 
        {
            clsids = new List<int>();
            labels = new List<string>();
            bboxs = new List<Rect>();
            scores = new List<float>();
        }
        public void add_data(int clsid, string label, Rect bbox, float score) 
        {
            clsids.Add(clsid);
            labels.Add(label);
            bboxs.Add(bbox);
            scores.Add(score);
        }
    };
    public class RTDETRProcess
    {
        public RTDETRProcess() { }
        /// <summary>
        /// The RTDETRProcess function processes an image with the specified target size, optional label path,
        /// threshold, and interpolation flags.
        /// </summary>
        /// <param name="target_size">The target_size parameter represents the size of the target. It is typically used to
        /// specify the width and height of an image or a region of interest.</param>
        /// <param name="label_path">The `label_path` parameter is a string that represents the path to a file
        /// containing labels.</param>
        /// <param name="threshold">The threshold parameter is a float value that determines the minimum
        /// confidence score required for an object detection result to be considered valid. Any detection
        /// result with a confidence score below the threshold will be ignored.</param>
        /// <param name="interpf">interpf is an enumeration that specifies the
        /// interpolation method used when resizing an image. It can have the following values:</param>
        public RTDETRProcess(Size target_size, string label_path = null, float threshold = 0.5f,
            InterpolationFlags interpf = InterpolationFlags.Linear)
        {
            this.target_size = target_size;
            this.threshold = threshold;
            this.interpf = interpf;
            if (label_path != null) { read_labels(label_path); }
      
        }
        /// <summary>
        /// The function preprocesses an input image by resizing it and converting it into a blob.
        /// </summary>
        /// <param name="image">The "image" parameter represents an image matrix. It is a data structure used in
        /// OpenCV to store and manipulate images.</param>
        /// <returns>
        /// The method is returning a Mat object.
        /// </returns>
        public Mat preprocess(Mat image)
        {
            im_shape = new List<float> { (float)image.Rows, (float)image.Cols };
            scale_factor = new List<float> { 640.0f / (float)image.Rows, 640.0f / (float)image.Cols};
            Mat input_mat = CvDnn.BlobFromImage(image, 1.0 / 255.0, target_size, 0, true, false);
            return input_mat;
        }
       /// <summary>
       /// The function takes in an array of scores and bounding box coordinates, and based on a flag,
       /// it either applies a threshold to filter the results or performs additional calculations to
       /// determine the maximum score and bounding box.
       /// </summary>
       /// <param name="score">An array of floating-point values representing the scores for each
       /// bounding box. The length of the array is 300.</param>
       /// <param name="bbox">The `bbox` parameter is an array of floats that represents the bounding
       /// box coordinates for each detected object. Each object is represented by 4 values in the
       /// array, which correspond to the x-coordinate, y-coordinate, width, and height of the bounding
       /// box.</param>
       /// <param name="post_flag">A boolean flag indicating whether to use a specific post-processing
       /// method or not.</param>
       /// <returns>
       /// The method is returning an object of type ResultData.
       /// </returns>
        public ResultData postprocess(float[] score, float[] bbox, bool post_flag)
        {
            ResultData result = new ResultData();
            if (post_flag)
            {
                for (int i = 0; i < 300; ++i)
                {
                    if (score[6 * i + 1] > threshold)
                    {
                        result.clsids.Add((int)score[6 * i]);
                        result.labels.Add(labels[(int)score[6 * i]]);
                        result.bboxs.Add(new Rect((int)score[6 * i + 2], (int)score[6 * i + 3],
                            (int)(score[6 * i + 4] - score[6 * i + 2]),
                            (int)(score[6 * i + 5] - score[6 * i + 3])));
                        result.scores.Add(score[6 * i + 1]);
                    }
                }
            }
            else
            {
                for (int i = 0; i < 300; ++i)
                {
                    float[] s = new float[80];
                    for (int j = 0; j < 80; ++j)
                    {
                        s[j] = score[80 * i + j];
                    }
                    int clsid = argmax(s, 80);
                    float max_score = sigmoid(s[clsid]);
                    if (max_score > threshold)
                    {
                        result.clsids.Add(clsid);
                        result.labels.Add(labels[clsid]);
                        float cx = (float)(bbox[4 * i] * 640.0 / scale_factor[1]);
                        float cy = (float)(bbox[4 * i + 1] * 640.0 / scale_factor[0]);
                        float w = (float)(bbox[4 * i + 2] * 640.0 / scale_factor[1]);
                        float h = (float)(bbox[4 * i + 3] * 640.0 / scale_factor[0]);
                        result.bboxs.Add(new Rect((int)(cx - w / 2), (int)(cy - h / 2), (int)w, (int)h));
                        result.scores.Add(max_score);
                    }
                }
            }
            return result;
        }
        /// <summary>
        /// The function "get_im_shape" returns a list of float values representing the shape of an
        /// image.
        /// </summary>
        /// <returns>
        /// A List of float values is being returned.
        /// </returns>
        public List<float> get_im_shape() { return im_shape; }
        /// <summary>
        /// The function "get_input_shape" returns a list of two floats representing the width and height of a
        /// target size.
        /// </summary>
        /// <returns>
        /// A list of two floats representing the width and height of the target size.
        /// </returns>
        public List<float> get_input_shape()
        {
            return new List<float> { (float)target_size.Width, (float)target_size.Height };
        }
        /// <summary>
        /// The function "get_scale_factor" returns a list of float values representing the scale
        /// factor.
        /// </summary>
        /// <returns>
        /// A List of float values is being returned.
        /// </returns>
        public List<float> get_scale_factor() { return scale_factor; }
        /// <summary>
        /// The function takes an input image and a set of results, and draws bounding boxes with labels
        /// and scores on the image.
        /// </summary>
        /// <param name="image">The "image" parameter represents an image matrix. It is a data structure
        /// used in OpenCV to store and manipulate images.</param>
        /// <param name="results">The "results" object contains the following information:</param>
        /// <returns>
        /// a modified version of the input image, with bounding boxes and labels drawn on it.
        /// </returns>
        public Mat draw_box(Mat image, ResultData results)
        {
            Mat re_image = image.Clone();
            //INFO("Infer result:");
            for (int i = 0; i < results.clsids.Count(); ++i)
            {
                int clsid = results.clsids[i];
                string label = results.labels[i];
                Rect bbox = results.bboxs[i];
                float score = results.scores[i];

                string score_str = score.ToString("0.0000");
                string text = label;
                text += ("  " + score_str);
                Cv2.Rectangle(re_image, bbox, new Scalar(255, 0, 0), 1);
                int y = 5;
                Size text_size = Cv2.GetTextSize(text, 0, 0.4, 1, out y);
                Rect rec = new Rect(bbox.TopLeft.X, bbox.TopLeft.Y - text_size.Height, 
                    text_size.Width, text_size.Height);
                List<Point> contour = new List<Point>();
                contour.Add(rec.TopLeft);
                contour.Add(new Point(rec.TopLeft.X + rec.Width, rec.TopLeft.Y));
                contour.Add(new Point(rec.TopLeft.X + rec.Width, rec.TopLeft.Y + rec.Height));
                contour.Add(new Point(rec.TopLeft.X, rec.TopLeft.Y + rec.Height));

                Cv2.FillConvexPoly(re_image, contour, new Scalar(0, 0, 0));
                Cv2.PutText(re_image, text, new Point(bbox.TopLeft.X, bbox.TopLeft.Y-2), 
                    HersheyFonts.HersheySimplex, 0.35, new Scalar(255, 255, 255), 1);

                //string msg = "  class_id : " + clsid.ToString() + ", label : " + label +
                //    ", confidence : " + score_str + ", left_top : [" + bbox.TopLeft.X.ToString("0.0000") 
                //    + ", " + bbox.TopLeft.X.ToString("0.0000") + "], right_bottom: [" +
                //    bbox.BottomRight.X.ToString("0.0000") + ", " + bbox.BottomRight.Y.ToString("0.0000") + "]";
                //INFO(msg);
            }
            return re_image;
        }


        /// <summary>
        /// The function reads labels from a file and stores them in a list.
        /// </summary>
        /// <param name="label_path">The parameter "label_path" is a string that represents the file path of the
        /// file containing the labels.</param>
        private void read_labels(string label_path)
        {
            labels = new List<string>();
            StreamReader stream_reader = new StreamReader(label_path);
            string item;
            while ((item = stream_reader.ReadLine()) != null)
            {
                labels.Add(item);
            }
            stream_reader.Close();
        }

        /// <summary>
        /// The sigmoid function takes in a float value and returns the result of applying the sigmoid function
        /// to that value.
        /// </summary>
        /// <param name="data">The parameter "data" is a float value that represents the input to the sigmoid
        /// function.</param>
        /// <returns>
        /// the result of the sigmoid function applied to the input data.
        /// </returns>
        private float sigmoid(float data)
        {
            return (float)(1.0f / (1 + Math.Exp(-data)));
        }

        /// <summary>
        /// The function `argmax` takes an array of floats and returns the index of the maximum value in the
        /// array.
        /// </summary>
        /// <param name="data">An array of float values.</param>
        /// <param name="length">The `length` parameter is the number of elements in the `data` array that
        /// should be considered for finding the maximum value.</param>
        /// <returns>
        /// The method is returning the index of the maximum value in the given array.
        /// </returns>
        private int argmax(float[] data, int length)
        {
            List<float> arr = new List<float>(data);
            float max = arr.Max();
            return arr.FindIndex(val => val == max);
        }

        private Size target_size;               // The model input size.
        private List<string> labels;    // The model classification label.
        private float threshold;                    // The threshold parameter.
        private InterpolationFlags interpf;     // The image scaling method.
        private List<float> im_shape;
        private List<float> scale_factor;
    }
}
