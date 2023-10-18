cmake_minimum_required(VERSION 3.15)

project(rtdetr-openvino VERSION 1.0 LANGUAGES CXX)

add_compile_options(-std=c++11)

set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 添加OpenCV搜索路径，替换成自己的OpenCV安装路径
list(APPEND CMAKE_PREFIX_PATH C:\\3rdpartylib\\opencv-4.5.5\\build\\x64\\vc15\\lib)

# 引入 OpenCV 库
find_package(OpenCV REQUIRED)
 
# 添加 OpenCV 库头文件搜索路径
include_directories(${OpenCV_INCLUDE_DIRS})


# OpenVINO 替换为自己的OpenVINO编译路径
set(OPENVINO_ROOT_PATH "C:\\Program Files (x86)\\Intel\\openvino_2023.1.0\\runtime")
set(OPENVINO_INCLUDE_DIRS ${OPENVINO_ROOT_PATH}/include)
set(OPENVINO_LIB ${OPENVINO_ROOT_PATH}/lib/intel64/Release/openvino.lib)

# OpenVINO 头文件
include_directories(${OPENVINO_INCLUDE_DIRS})

# 将生成的可执行文件保存到指定路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./")

# 编译成可执行文件
add_executable(rt-detr_openvino_cpp main.cpp rtdert_predictor.cpp process.cpp)

target_include_directories(rt-detr_openvino_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(rt-detr_openvino_cpp PRIVATE ${OPENVINO_LIB} ${OpenCV_LIBS} )