using System;
using System.Runtime.InteropServices;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using OpenVinoSharp;
using static System.Net.Mime.MediaTypeNames;
using static rt_detr_time_text.Msg;
namespace rt_detr_time_text
{
    internal class Program
    {
        static void Main(string[] args)
        {
            rtdetr();
            //yolov8();

        }
        static void rtdetr() 
        {
            INFO("Hello, World!");
            string model_path = "E:\\Model\\RT-DETR\\RTDETR\\rtdetr_r34vd_6x_coco.xml";
            string image_path = "E:\\GitSpace\\RT-DETR-OpenVINO\\image\\000000570688.jpg";
            string label_path = "E:\\GitSpace\\RT-DETR-OpenVINO\\image\\COCO_lable.txt";
            Mat image = Cv2.ImRead(image_path);
            Mat result_mat = new Mat();
            int n = 100;
            for (int i = 0; i < n; ++i) {
                INFO("Model predict: " + i.ToString());
                RTDETRPredictor predictor = new RTDETRPredictor(model_path, label_path, "CPU", true);
                result_mat = predictor.predict(image);
            }
            INFO((RTDETRPredictor.load_model/n).ToString("0.00"));
            INFO((RTDETRPredictor.process_image / n).ToString("0.00"));
            INFO((RTDETRPredictor.load_data / n).ToString("0.00"));
            INFO((RTDETRPredictor.infer / n).ToString("0.00"));
            INFO((RTDETRPredictor.postprocess / n).ToString("0.00"));

            //Console.WriteLine("Hello, World!");
            //INFO("Hello, World!");
            //string label_path = "E:\\GitSpace\\RT-DETR-OpenVINO\\image\\COCO_lable.txt";
            //string video_path = "E:\\ModelData\\human.mp4";
            //string model_path = "E:\\Model\\RT-DETR\\RTDETR\\rtdetr_r34vd_6x_coco.onnx";
            //VideoCapture video = new VideoCapture(video_path);
            //RTDETRPredictor predictor = new RTDETRPredictor(model_path, label_path, "CPU", true);
            //Mat result_mat = new Mat();
            //Mat image = new Mat();
            //// 创建视频保存器
            //VideoWriter video_writer = new VideoWriter(@"output.avi",
            //    FourCC.MP42, 30, new Size(video.FrameWidth, video.FrameHeight));
            //while (video.Read(image))
            //{
            //    result_mat = predictor.predict(image);
            //    Cv2.ImShow("video", result_mat);
            //    Cv2.WaitKey(1);
            //    video_writer.Write(result_mat);
            //}
            //video_writer.Release();
        }

        static void yolov8() 
        {
            string model_path = "D:\\yolov8m.onnx";
            string video_path = "E:\\ModelData\\human.mp4";
            // -------- Step 1. Initialize OpenVINO Runtime Core --------
            Core core = new Core();
            // -------- Step 2. Read a model --------
            Console.WriteLine("[INFO] Loading model files: {0}", model_path);
            Model model = core.read_model(model_path);

            // -------- Step 3. Loading a model to the device --------
            CompiledModel compiled_model = core.compile_model(model, "CPU");

            // -------- Step 4. Create an infer request --------
            InferRequest infer_request = compiled_model.create_infer_request();
            // -------- Step 5. Process input images --------
            VideoCapture video = new VideoCapture(video_path);
            Mat result_mat = new Mat();
            Mat image = new Mat();

            while (video.Read(image))
            {
                int max_image_length = image.Cols > image.Rows ? image.Cols : image.Rows;
                Mat max_image = Mat.Zeros(new OpenCvSharp.Size(max_image_length, max_image_length), MatType.CV_8UC3);
                Rect roi = new Rect(0, 0, image.Cols, image.Rows);
                image.CopyTo(new Mat(max_image, roi));
                float[] factors = new float[4];
                factors[0] = factors[1] = (float)(max_image_length / 640.0);
                factors[2] = image.Rows;
                factors[3] = image.Cols;

                // -------- Step 6. Set up input --------
                Tensor input_tensor = infer_request.get_input_tensor();
                Shape input_shape = input_tensor.get_shape();
                Mat input_mat = CvDnn.BlobFromImage(max_image, 1.0 / 255.0, new Size(input_shape[2], input_shape[3]), 0, true, false);
                float[] input_data = new float[input_shape[1] * input_shape[2] * input_shape[3]];
                Marshal.Copy(input_mat.Ptr(0), input_data, 0, input_data.Length);
                input_tensor.set_data<float>(input_data);


                // -------- Step 7. Do inference synchronously --------

                DateTime start = DateTime.Now;
                infer_request.infer();
                DateTime end = DateTime.Now;
                INFO((end - start).TotalMilliseconds.ToString());
            }
        }
    }
}