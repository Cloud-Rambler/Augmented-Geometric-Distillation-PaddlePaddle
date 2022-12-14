# 论文名称

## 目录

- [1. 简介]()
- [2. 数据集和复现精度]()
- [3. 准备数据与环境]()
    - [3.1 准备环境]()
    - [3.2 准备数据]()
    - [3.3 准备模型]()
- [4. 开始使用]()
    - [4.1 模型训练]()
    - [4.2 模型评估]()
    - [4.3 模型预测]()
- [5. 模型推理部署]()
    - [5.1 基于Inference的推理]()
    - [5.2 基于Serving的服务化部署]()
- [6. 自动化测试脚本]()
- [7. LICENSE]()
- [8. 参考链接与文献]()


**注意：**

(1) 目录可以使用[gh-md-toc](https://github.com/ekalinin/github-markdown-toc)生成；

(2) 示例repo和文档可以参考：[AlexNet_paddle](https://github.com/littletomatodonkey/AlexNet-Prod/blob/tipc/pipeline/Step5/AlexNet_paddle/README.md)。

## 1. 简介

简单的介绍模型，以及模型的主要架构或主要功能，如果能给出效果图，可以在简介的下方直接贴上图片，展示模型效果。然后另起一行，按如下格式给出论文名称及链接、参考代码链接、aistudio体验教程链接。

注意：在给出参考repo的链接之后，建议添加对参考repo的开发者的致谢。

<div align="center">
<img src="./images/pipeline.png"  width = "600" />
<p>AGD整体流程图</p>
</div>

**论文:** [Augmented Geometric Distillation for Data-Free Incremental Person ReID](https://openaccess.thecvf.com/content/CVPR2022/html/Lu_Augmented_Geometric_Distillation_for_Data-Free_Incremental_Person_ReID_CVPR_2022_paper.html)



在此非常感谢`$eddiely$`等人贡献的[Augmented-Geometric-Distillation](https://github.com/eddielyc/Augmented-Geometric-Distillation)，提高了本repo复现论文的效率。



## 2. 数据集和复现精度

### 2.1 数据集
数据集为MSMT17和Market1501，数据集可以在AI Studio数据集页面上下载。

### 2.2 复现精度
| 任务  | 数据集| 论文mAP|复现mAP|论文R@1 | 复现R@1  |  
| :--- | :--- |  :----:  | :--------: |  :----  |   :----  | 
|任务一|msmt17| 46.5%|   46.2%|72.1% |70.8%  |

| 任务  | 数据集    | 论文mAP | 复现mAP | 论文R@1 | 复现R@1  |
|:----|:-------|:-----:|:-----:|:------|:------| 
| 任务二 | msmt17 | 41.9% | 39.6% | 67.5% | 65.3% |
| 任务二 | market | 80.5% | 80.4% | 91.9%  | 91.5%   |






## 3. 准备数据与环境


### 3.1 准备环境

* 安装环境

```bash
conda create -n ppcls python=3.7
pip install -r requirements.txt
```

* 下载代码

```bash
git clone https://github.com/Cloud-Rambler/Augmented-Geometric-Distillation-PaddlePaddle.git
cd Augmented-Geometric-Distillation-PaddlePaddle
```

### 3.2 准备数据

将数据集解压到`./data`目录下，数据集目录结构如下：

Datasets Structure
```
./data
- market
  - bounding_box_test
  - bounding_box_train
  - query
 
- msmt17
  - bounding_box_test
  - bounding_box_train
  - query

...
```

### 3.3 准备模型


前往[这里](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/others/resnet50-19c8e357_torch2paddle.pdparams)下载预训练模型并放入`./checkpoints`目录下。


## 4. 开始使用


### 4.1 模型训练

* 训练任务$T_1$ (MSMT17):
```
python ./tools/main.py -g 0 --dataset msmt17 --logs-dir ./logs/msmt17
```

* 通过DeepInversion[1]生成dreaming data:
```
python ./tools/inversion.py -g 0 --generation-dir ./data/generations_r50_msmt17 --shots 40 --iters 640 --teacher ./logs/msmt17
```

* 使用 Geometric Distillation loss在任务$T_2$上训练 incremental model
```
python ./tools/main_incremental.py --dataset market --previous ./logs/msmt17 --logs-dir ./logs/msmt17-market_GD --inversion-dir ./data/generations_r50_msmt17 -g 0 --evaluate 80 --seed 1 --algo-config ./ppcls/configs/res-triangle.yaml
```

* 使用 simple Geometric Distillation loss (detailed in Supp. and usually report better performance)在任务$T_2$上训练 incremental model:
```
python ./tools/main_incremental.py --dataset market --previous ./logs/msmt17 --logs-dir ./logs/msmt17-market_simGD --inversion-dir ./data/generations_r50_msmt17 -g 0 --evaluate 80 --seed 1 --algo-config ./ppcls/configs/sim-res-triangle.yaml
```

* 使用Augmented Distillation在任务$T_2$上训练 incremental model:
```
python .tools/main_incrementalX.py --dataset market --previous ./logs/msmt17 --logs-dir ./logs/msmt17-market_XsimGD --inversion-dir ./data/generations_r50_msmt17 -g 0 --evaluate 80 --seed 1 --peers 2 --epoch 80 --algo-config ./ppcls/configs/inverXion.yaml
```

### 4.2 模型评估

```angular2svg
python ./tools/evaluate.py --dataset msmt17 market --ckpt ./logs/msmt17-market_XsimGD/checkpoint.pdparams --output 
```

## 5. TIPC



* 准备数据

```bash
# 解压数据，如果您已经解压过，则无需再次运行该步骤
cd data
tar -xf lite_data.tar
```

* 运行测试命令

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/AGD/train_infer_python.txt lite_train_lite_infer
```

如果运行成功，在终端中会显示下面的内容，具体的日志也会输出到`test_tipc/output/`文件夹中的文件中。

```
Run successfully with command - python ./tools/main_incrementalX.py --dataset market --previous ./logs/msmt17 --logs-dir ./logs/msmt17-market_XsimGD --inversion-dir ./data/generations_r50_msmt17 -g 0 --evaluate 80 --seed 1 --peers 2 --algo-config ./ppcls/configs/inverXion.yaml   !  
 ...
Run successfully with command - python ./tools/evaluate.py --dataset msmt17 market --ckpt ./logs/msmt17-market_XsimGD/checkpoint.pdparams --output  !
```


## 6. LICENSE

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。

## 7. 参考链接与文献
[1] Yin, Hongxu, et al. "Dreaming to distill: Data-free knowledge transfer via deepinversion." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
