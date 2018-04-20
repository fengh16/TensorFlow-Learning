# 环境

Windows 10, GTX 950M

腾讯云辣鸡学生机（ubuntu）……反正只放了一个公众号后台，跑跑tensorflow也无妨，大不了跑个几天几夜

# 安装

安装python3, pip等，之后ubuntu系统直接`pip install tensorflow`，win10下面……一定要下载**64位**的python 3.5/3.6（我也不知道为啥原来下载的是32位的）！之后直接`pip install tensorflow`。如果有报错，试试`pip install --upgrade --ignore-installed tensorflow`

如果安装gpu版本，那就折腾吧……（原始教程：https://blog.csdn.net/vcvycy/article/details/79298703）

* 安装cuda 9.1+VS2017
* 安装cudnn7.0
* 复制一堆文件
  * Copy \cuda\bin\cudnn64_7.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin.
  * Copy \cuda\ include\cudnn.h to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include.
  * Copy \cuda\lib\x64\cudnn.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64.
* 到https://github.com/fo40225/tensorflow-windows-wheel下载Cuda9.1版本的tensorflow

