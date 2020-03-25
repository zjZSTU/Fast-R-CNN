# Fast R-CNN

[![Documentation Status](https://readthedocs.org/projects/fast-r-cnn/badge/?version=latest)](https://fast-r-cnn.readthedocs.io/zh_CN/latest/?badge=latest) [![standard-readme compliant](https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme) [![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org) [![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)

> `Fast R-CNN`算法实现

学习论文[Fast R-CNN](https://arxiv.org/abs/1504.08083)，实现`Fast R-CNN`算法，完成目标检测器的训练和使用

<!-- `Fast R-CNN`实现由如下`3`部分组成：
 
 1. 区域建议算法（`SelectiveSearch`）
 2. 卷积网络模型（`AlexNet`）
 3. 线性分类器（线性`SVM`） -->

## 内容列表

- [背景](#背景)
- [安装](#安装)
- [用法](#用法)
- [主要维护人员](#主要维护人员)
- [参与贡献方式](#参与贡献方式)
- [许可证](#许可证)

## 背景

`Fast R-CNN`在`R-CNN`的基础上进一步发展，能够实现更快的训练和检测

>本文提出一种快速的基于区域的卷积神经网络目标检测方法（Fast R-CNN）。Fast R-CNN在之前工作的基础上，利用深度卷积神经网络对目标方案进行有效分类。与以往的工作相比，Fast R-CNN在提高训练和测试速度的同时，也提高了检测精度。Fast R-CNN使用VGG16网络，训练速度比R-CNN快9倍，测试速度快213倍，同时在PASCAL VOC 2012上实现了更高的mAP。和SPPnet相比，Fast R-CNN使用VGG16网络，训练速度快3倍，测试速度快10倍，同时更加精确

## 安装

### 本地编译文档

需要预先安装以下工具：

```
$ pip install mkdocs
```

## 用法

### 文档浏览

有两种使用方式

1. 在线浏览文档：[Fast R-CNN](https://fast-r-cnn.readthedocs.io/zh_CN/latest/)

2. 本地浏览文档，实现如下：

    ```
    $ git clone https://github.com/zjZSTU/Fast-R-CNN.git
    $ cd Fast-R-CNN
    $ mkdocs serve
    ```
    启动本地服务器后即可登录浏览器`localhost:8000`

## 主要维护人员

* zhujian - *Initial work* - [zjZSTU](https://github.com/zjZSTU)

## 参与贡献方式

欢迎任何人的参与！打开[issue](https://github.com/zjZSTU/Fast-R-CNN/issues)或提交合并请求。

注意:

* `GIT`提交，请遵守[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)规范
* 语义版本化，请遵守[Semantic Versioning 2.0.0](https://semver.org)规范
* `README`编写，请遵守[standard-readme](https://github.com/RichardLitt/standard-readme)规范

## 许可证

[Apache License 2.0](LICENSE) © 2020 zjZSTU
