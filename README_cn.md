![OpenVINO™ C# API](https://socialify.git.ci/guojin-yan/OpenVINO-CSharp-API/image?description=1&descriptionEditable=💞%20OpenVINO%20wrapper%20for%20.NET💞%20&forks=1&issues=1&logo=https%3A%2F%2Fs2.loli.net%2F2023%2F01%2F26%2FylE1K5JPogMqGSW.png&name=1&owner=1&pattern=Circuit%20Board&pulls=1&stargazers=1&theme=Light)

简体中文 |[English](README.md) 

# RT-DETR-OpenVINO

This project mainly demonstrates the deployment of RT-DETR model cases based on OpenVINO C++, Python, and C # API.

本项目主要基于Windows环境展示了基于OpenVINO C++、Python和C#API的RT-DETR模型案例的部署。

# 🛠 项目环境

| Python 环境                                                  | C++环境                             | C# 环境                                                      |
| ------------------------------------------------------------ | :---------------------------------- | :----------------------------------------------------------- |
| paddlepaddle=2.5.1<br/>onnx=1.13.0 <br/>paddle2onnx=0.5 <br/>paddledet <br/>opencv-python=4.8.1.78 <br/>openvino=2023.1.0 <br/>pillow=10.0.1 | opencv=4.5.5 <br/>openvino=2023.1.0 | OpenCvSharp4.Windows=4.8.0.20230708 <br/>OpenVINO.CSharp.win=3.1.1 |

# 🎯 模型下载与转换

## ♻ 环境安装

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

## ➿ 模型导出

```shell
cd PaddleDetection
python tools/export_model.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_6x_coco.pdparams trt=True --output_dir=output_inference
```

<div align=center><span><img src="https://s2.loli.net/2023/10/18/bwBfI3JR7goH5Da.png" height=300/></span></div>

上图为我们导出的RT-DETR模型，该模型实际是包含后处理的，因此模型的输入有三个节点，如果大家感觉使用较为麻烦，也可以导出不加后处理的模型，实现方式如下：

修改RT-DETR模型的配置文件，配置文件路径为：``.\PaddleDetection\configs\rtdetr\_base_\rtdetr_r50vd.yml``，在配置文件DETR项目下增加``exclude_post_process: True``语句。

<div align=center><span><img src="https://s2.loli.net/2023/10/18/tA2JFsqaR3L6Vnm.png" height=300/></span></div>

然后重新运行模型导出指令，便可以获取不包含后处理的模型，如下图所示：

<div align=center><span><img src="https://s2.loli.net/2023/10/18/OkWv5EcipdwrI7D.png" height=300/></span></div>

## 🔮 转换ONNX格式

```shell
paddle2onnx --model_dir=./output_inference/rtdetr_r50vd_6x_coco/ --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 16 --save_file rtdetr_r50vd_6x_coco.onnx
```

## 🎞️ 转换IR格式

目前我们所使用的Paddle所导出来的模型为动态形状，并且OpenVINO支持动态模型输入，但是为了防止后续处理时方便，此处我们在导出IR模型时，对模型的形状进行固定，通过以下指令便可以实现：

```shell
ovc rtdetr_r50vd_6x_coco.onnx --input “image[1,3,640,640], im_shape[1,2], scale_factor[1,2]”
```

如果是前文中导出来的不带后处理的模型，转换指令为：

```shell
ovc rtdetr_r50vd_6x_coco.onnx --input image[1,3,640,640]
```

# 🗃️RT-DETR INT8 量化

如果想实现RT-DETR INT8 量化，可以参考以下文章的步骤实现：[**Convert and Optimize RT-DETR  real-time object detection with OpenVINO™**](./optimize/openvino-convert-and-optimize-rt-detr.ipynb)

# 🎨 案例测试

## 😇 Python

```
git clone https://github.com/guojin-yan/RT-DETR-OpenVINO.git
cd RT-DETR-OpenVINO/scr/python
python main.py [model path] [image path] [label path] [post flag(1/0)]
```

- \[model path]：表示预测模型地址，可以根据上文步骤进行导出，或者下载本仓库中发布的模型。
- [image path]：表示待预测图片地址，文件位置在``RT-DETR-OpenVINO\image``路径下。
- [label path]：表示预测结果类别文件，文件位置在``RT-DETR-OpenVINO\image``路径下。
- [post flag(1/0)]：表示是否包含后处理的模型，post_flag=0表示不包含后处理，post_flag=1表示包含后处理.



| Console Output                                               | Result Image                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <span><img src="https://s2.loli.net/2023/10/18/nZGydeSl9NWD54P.png" height=400/></span> | <span><img src="https://s2.loli.net/2023/10/18/ZgFi2tzX3bvHc1y.png" height=400/></span> |

## 🥰 C++

```
git clone https://github.com/guojin-yan/RT-DETR-OpenVINO.git
cd RT-DETR-OpenVINO/scr/cpp
```

C++ 案例中使用了Cmake编译，若要成功编译该项目，需要根据自己电脑安装对应的依赖库，该项目需要安装OpenVINO以及OpenCV；安装之后，修改``RT-DETR-OpenVINO\src\cpp\CMakeLists.txt``文件中OpenVINO以及OpenCV的编译路径地址即可。

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

如果您准备使用OpenVINO部署RT-DETR模型，欢迎参考本案例。在使用中有任何问题，可以通过以下方式与我联系。

<div align=center><span><img src="https://s2.loli.net/2023/10/18/d6QUWL7HG523BuR.png" height=300/></span></div>
