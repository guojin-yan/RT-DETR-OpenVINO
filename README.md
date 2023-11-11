![OpenVINO™ C# API](https://socialify.git.ci/guojin-yan/OpenVINO-CSharp-API/image?description=1&descriptionEditable=💞%20OpenVINO%20wrapper%20for%20.NET💞%20&forks=1&issues=1&logo=https%3A%2F%2Fs2.loli.net%2F2023%2F01%2F26%2FylE1K5JPogMqGSW.png&name=1&owner=1&pattern=Circuit%20Board&pulls=1&stargazers=1&theme=Light)

[简体中文](README_cn.md) |English

# RT-DETR-OpenVINO

This project mainly demonstrates the deployment of RT-DETR model cases based on OpenVINO C++, Python, and C # API.

# 🛠 Project Environment

| Python Environment                                           | C++ Environment                     | C# Environment                                               |
| ------------------------------------------------------------ | :---------------------------------- | :----------------------------------------------------------- |
| paddlepaddle=2.5.1<br/>onnx=1.13.0 <br/>paddle2onnx=0.5 <br/>paddledet <br/>opencv-python=4.8.1.78 <br/>openvino=2023.1.0 <br/>pillow=10.0.1 | opencv=4.5.5 <br/>openvino=2023.1.0 | OpenCvSharp4.Windows=4.8.0.20230708 <br/>OpenVINO.CSharp.win=3.1.1 |

# 🎯 Model Download and Cconversion

## ♻ Environmental Installation

```shell
# Creating a virtual environment using Conda.
conda create -n paddledet python=3.10
conda activate paddledet

# Install paddlepaddle
python -m pip install paddlepaddle==2.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Clone PaddleDetection repository
$ git clone https://github.com/PaddlePaddle/PaddleDetection.git
$ cd PaddleDetection
$ git checkout develop

# Compile and install paddledet
$ pip install -r requirements.txt
$ python setup.py install

# Convert ONNX format environment
pip install onnx==1.13.0
pip install paddle2onnx==1.0.5

# Install OpenVINO.
# Convert IR format environment and deploy model in Python environment
pip install openvino==2023.1.0
```

## ➿ Model Export

```shell
cd PaddleDetection
python tools/export_model.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_6x_coco.pdparams trt=True --output_dir=output_inference
```

<div align=center><span><img src="https://s2.loli.net/2023/10/18/bwBfI3JR7goH5Da.png" height=300/></span></div>

The above figure shows our exported RT-DETR model, which actually includes post-processing. Therefore, the input of the model has three nodes. If you find it inconvenient to use, you can also export a model without post-processing. The implementation method is as follows:
Modify the configuration file of the RT-DETR model, with the path to the configuration file:``.\PaddleDetection\configs\rtdetr\_base_\rtdetr_r50vd.yml``,  add 'exclude' under the DETR project in the configuration file_ Post_ Process: ``exclude_post_process: True``.

<div align=center><span><img src="https://s2.loli.net/2023/10/18/tA2JFsqaR3L6Vnm.png" height=300/></span></div>

Then rerun the model export command to obtain the model without post-processing, as shown in the following figure:

<div align=center><span><img src="https://s2.loli.net/2023/10/18/OkWv5EcipdwrI7D.png" height=300/></span></div>

## 🔮 Convert ONNX Format

```shell
paddle2onnx --model_dir=./output_inference/rtdetr_r50vd_6x_coco/ --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 16 --save_file rtdetr_r50vd_6x_coco.onnx
```

## 🎨 Convert IR Format

At present, the model exported by Paddle we are using is a dynamic shape, and OpenVINO supports dynamic model input. However, to prevent convenience in subsequent processing, we fix the shape of the model when exporting the IR model. This can be achieved by using the following instructions:

```shell
ovc rtdetr_r50vd_6x_coco.onnx --input “image[1,3,640,640], im_shape[1,2], scale_factor[1,2]”
```

If it is a model without post-processing exported from the previous text, the conversion instruction is:

```shell
ovc rtdetr_r50vd_6x_coco.onnx --input image[1,3,640,640]
```

# 🗃️RT-DETR INT8 Quantization

If you want to achieve RT-DETR INT8 quantization, you can refer to the steps in the following article to implement it: [**Convert and Optimize RT-DETR  real-time object detection with OpenVINO™**](./optimize/openvino-convert-and-optimize-rt-detr.ipynb)

# 🎨 Case Testing

## 😇 Python

```
git clone https://github.com/guojin-yan/RT-DETR-OpenVINO.git
cd RT-DETR-OpenVINO/scr/python
python main.py [model path] [image path] [label path] [post flag(1/0)]
```

- \[model path]：Represents the address of the prediction model, which can be exported according to the steps above or downloaded from the model published in this warehouse.
- [image path]：Indicates the address of the image to be predicted, and the file location is in the ``RT-DETR-OpenVINO\image`` path.
- [label path]：Represent the prediction result category file,  and the file location is in the ``RT-DETR-OpenVINO\image`` path.
- [post flag(1/0)]：Indicates whether the model includes post-processing, post_ flag=0 indicates no post-processing, post_ flag=1 indicates the inclusion of post-processing



| Console Output                                               | Result Image                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <span><img src="https://s2.loli.net/2023/10/18/nZGydeSl9NWD54P.png" height=400/></span> | <span><img src="https://s2.loli.net/2023/10/18/ZgFi2tzX3bvHc1y.png" height=400/></span> |

## 🥰 C++

```
git clone https://github.com/guojin-yan/RT-DETR-OpenVINO.git
cd RT-DETR-OpenVINO/scr/cpp
```

C++ 案例中使用了Cmake编译，若要成功编译该项目，需要根据自己电脑安装对应的依赖库，该项目需要安装OpenVINO以及OpenCV；安装之后，修改`文件中OpenVINO以及OpenCV的编译路径地址即可。

In the C++case, Cmake compilation was used. To successfully compile this project, it is necessary to install the corresponding dependency libraries based on one's own computer. This project requires the installation of OpenVINO and OpenCV; After installation, modify the compilation path addresses of OpenVINO and OpenCV in the ``RT-DETR-OpenVINO\src\cpp\CMakeLists.txt`` file.

```
mkdir build && cd build
cmake ..
make
rt-detr_openvino_cpp.exe [model path] [image path] [label path] [post flag(1/0)]
```

| Console Output                                               | Result Image                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <span><img src="https://s2.loli.net/2023/10/18/XeONfYJdmWSKMZQ.png" height=400/></span> | <span><img src="https://s2.loli.net/2023/10/18/FpMunTeOKXvidjI.png" height=400/></span> |

## 😀 C#

```
git clone https://github.com/guojin-yan/RT-DETR-OpenVINO.git
cd RT-DETR-OpenVINO/scr/csharp
dotnet run [model path] [image path] [label path] [post flag(1/0)]
```

| Console Output                                               | Result Image                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <span><img src="https://s2.loli.net/2023/10/18/IK4ZnPFHBTNEi1X.png" height=400/></span> | <span><img src="https://s2.loli.net/2023/10/18/KifdwrtRJ2UIcBQ.png" height=400/></span> |

# 📱 Contact 

If you are planning to deploy the RT-DETR model using OpenVINO, please refer to this case. If you have any questions during use, you can contact me through the following methods.

<div align=center><span><img src="https://s2.loli.net/2023/10/18/d6QUWL7HG523BuR.png" height=300/></span></div>
