# 基于同态加密隐私保护的联邦学习系统 (FLHE)

## 项目概述

该项目使用 Flask 作为 Web 服务器，实现了一种结合同态加密技术的联邦学习解决方案。用户可以提交请求以运行带有可自定义参数的演示模型，支持多种数据集和安全设置。

## 特性

- 支持多个数据集：MNIST、CIFAR-10、FashionMNIST。
- 可选择不同的深度学习模型：CNN、ResNet18。
- 实现同态加密方法：CKKS 和 Paillier。
- 提供模型训练过程的实时日志记录。
- 前后端交互。

## python版本

确保您已安装 Python 3.9，以及所需的库。具体请查看 `requirements.txt` 文件。

## 目录结构
    federated-learning-he/
        └── front_end/   前端代码
        └── models/ckks.py ：ckks相关代码
        └── models/ckks.py ：ckks相关代码
        └── models/Nets.py：模型代码
        └── models/test.py：测试全局模型代码
        └── models/Update.py: 训练代码
        └── roles/TA.py：分发密钥
        └── utils/options.py 参数设定
        └── utils/sampling.py数据分布代码
        └── main.py 联邦学习流程的主代码
        └── server.py 后端服务器代码
  

## 安装步骤

1. 克隆代码库：
   ```bash  
   git clone <https://github.com/SWORDandPOWER/federated-learning-he.git>  
   cd <federated-learning-he>

2. 创建虚拟环境：
    ```bash  
   conda create -n <envname>  python=3.9
   conda avtivate envname

3. 安装必要库
   ```bash  
   pip install -r requirements.txt
   

## 运行应用程序

1. 启动 Flask 服务器：
   ```bash  
   python server.py
2. 服务器将在 http://localhost:5000 上运行。

## API 接口示例

### 开始训练接口

- **接口：** `/run-demo`
- **方法：** `POST`
- **请求格式：**
   ```json  
   {  
     "dataset": "mnist",       // 选项：mnist, cifar, fmnist  
     "security": "ckks",       // 选项：ckks, paillier, no  
     "model": "cnn",           // 选项：cnn, resnet18  
     "epochs": 10,             // （可选）训练的轮数  
     "iid": true,              // （可选）布尔值，表示 IID 分布  
     "num_users": 5,           // （可选）用户数量  
     "frac": 0.1               // （可选）参与训练的用户比例  
   }
- **响应**
  ```json  
  {
     "code":"200 OK"
     "status": "started",
     "message": "已启动！请到后端终端查看实时日志。"
  }
