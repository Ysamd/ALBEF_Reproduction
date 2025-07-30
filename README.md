ALBEF
ALBEF

# ALBEF模型复现全流程指南 :smile:


 共分为以下步骤：
 - 下载源代码 (==建议使用最新官方Release，以免版本不兼容==)
 - 下载环境依赖
 - 下载数据集
 - 利用官方预训练权重进行训练
 - 对下游任务进行微调评估
 - 对比结果
 - 其他问题解决
 
 ## 下载源代码
 可在 ALBEF 的[官方 GitHub](https://github.com/salesforce/ALBEF)仓库找到源代码
 ==注意：请完整下载，不要漏文件==
 
 ## 下载环境依赖
 除了官方的Readme文档要求的：
 - Pytorch 1.8.0
 - transformers 4.8.1
 - timm 0.4.9
 ```python
 pip install pytorch==1.8.0
 pip install transformers==4.8.1
 pip install timm==0.4.9
 ```
 之外还有：
 - 创建一个新环境
 - 建议python版本为3.8
 ```python
conda create -n albef python=3.8
 ```
 - matplotlib (可装可不装，后续可视化能用到)
 ```python
 pip install matplotlib
 ```
 ## 下载数据集
 - [Flickr30k数据集](http://shannon.cs.illinois.edu/DenotationGraph/)
   需要在里面填表，然后会发下载网址到你的邮箱，直接[点这里](http://shannon.cs.illinois.edu/DenotationGraph/data/index.html)也可以，但是下载可能不稳定，打开后下载前两个就可以了
- [MSCOCO数据集]( https://cocodataset.org/#download)
  ALBEF用的是2014的，并且是用val2014的前5000张图片作为测试集
  ==注意：这里会弹出不安全下载，复制一下链接，在新的标签页打开就好了，再点一下保留(以Chrome为例)==
 - [refcoco数据集](https://huggingface.co/datasets/jxu124/refcoco)
   由于官方并没有给出refcoco的直接下载接口，所以只能通过HuggingFace下载，里面的数据可能需要做相应预处理，才能符合ALBEF的格式
  - [VQA数据集](https://visualqa.org/download.html)
    这里用的是2.0，与COCO2014的图片对应，需要将问题和答案与图像路径相对应
- [NLVR2数据集](https://github.com/lil-lab/nlvr/tree/master/nlvr2)
   这里有下载图片的脚本，复制到本地直接运行即可
最后，这里的目录结构应该是：
data/
├── coco/
│   ├── train2014/
│   ├── val2014/
│   └── annotations/
├── vqa/
│   ├── Questions/
│   └── Annotations/
├── refcoco/
│   ├── refs(unc).json
│   ├── refs(unc+).json
│   ├── refs(google).json
│   ├── train_images/
│   └── val_images/
├── nlvr2/
│  ├── images/
│  ├── train.json
│  ├── val.json
│└── test.json
├──  flickr30k/ 
 │ ├── images/
 │ ├── train.json
 │ ├── val.json
 │└── test.json
 ==注意：这里的Annotations存的也是这些json文件，也就是标注信息==
 ## 利用官方预训练权重进行下游任务微调与评估
 在开始训练之前，最好在huggingface上手动下载一个bert-base-uncased的模型权重和配置文件，因为如果自动下载的话，可能出现网络问题，[下载地址](https://huggingface.co/google-bert/bert-base-uncased)
 必要的文件是：
 - pytroch_model.bin
 - tokenizer.json
 - tokenizer_config.json
 - config.json
 - vocab.txt
 在你的代码中需要改一下路径
 ```python
 from transformers import BertModel

model = BertModel.from_pretrained("/path/to/local/bert-base-uncased")
```
 下载官方的预训练权重（这些在官方的Readme里能看到）
 [14M](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth)
 [4M](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF_4M.pth)
开始训练（以Flickr30k为例子）
```python
python Retrieval.py \
  --config ./configs/Retrieval_flickr.yaml \
  --output_dir output/Retrieval_flickr \
  --textencoder ./pretrained/bert-base-uncased \
  --checkpoint ./pretrained/ALBEF_14M.pth \
  --device cuda \
  --distributed False
```
==这里如果多卡训练，distributed请设置为True==
评估（以Flickr30k为例子）
```python
python Retrieval.py \
  --config ./configs/Retrieval_flickr.yaml \
  --output_dir output/Retrieval_flickr \
  --textencoder ./pretrained/bert-base-uncased \
  --checkpoint output/Retrieval_flickr/checkpoint_best.pth \
  --evaluate
```

## 对比结果
ALBEF的[官方论文](https://arxiv.org/pdf/2107.07651)地址
具体指标详见论文

可视化内容不再做赘述

## 其他问题解决
- 如遇到GPU显存不足，请自行降低图片分辨率或是降低batchsize的大小（在config的文件夹里）





ALBEF模型复现全流程指南 😄
共分为以下步骤：

下载源代码 (建议使用最新官方Release，以免版本不兼容)
下载环境依赖
下载数据集
利用官方预训练权重进行训练
对下游任务进行微调评估
对比结果
其他问题解决
下载源代码
可在 ALBEF 的官方 GitHub仓库找到源代码
注意：请完整下载，不要漏文件

下载环境依赖
除了官方的Readme文档要求的：

Pytorch 1.8.0
transformers 4.8.1
timm 0.4.9
pip install pytorch==1.8.0
pip install transformers==4.8.1
pip install timm==0.4.9
之外还有：

创建一个新环境
建议python版本为3.8
conda create -n albef python=3.8
matplotlib (可装可不装，后续可视化能用到)
pip install matplotlib
下载数据集
Flickr30k数据集
需要在里面填表，然后会发下载网址到你的邮箱，直接点这里也可以，但是下载可能不稳定，打开后下载前两个就可以了
MSCOCO数据集
ALBEF用的是2014的，并且是用val2014的前5000张图片作为测试集
注意：这里会弹出不安全下载，复制一下链接，在新的标签页打开就好了，再点一下保留(以Chrome为例)
refcoco数据集
由于官方并没有给出refcoco的直接下载接口，所以只能通过HuggingFace下载，里面的数据可能需要做相应预处理，才能符合ALBEF的格式
VQA数据集
这里用的是2.0，与COCO2014的图片对应，需要将问题和答案与图像路径相对应
NLVR2数据集
这里有下载图片的脚本，复制到本地直接运行即可
最后，这里的目录结构应该是：
data/
├── coco/
│ ├── train2014/
│ ├── val2014/
│ └── annotations/
├── vqa/
│ ├── Questions/
│ └── Annotations/
├── refcoco/
│ ├── refs(unc).json
│ ├── refs(unc+).json
│ ├── refs(google).json
│ ├── train_images/
│ └── val_images/
├── nlvr2/
│ ├── images/
│ ├── train.json
│ ├── val.json
│└── test.json
├── flickr30k/
│ ├── images/
│ ├── train.json
│ ├── val.json
│└── test.json
注意：这里的Annotations存的也是这些json文件，也就是标注信息
利用官方预训练权重进行下游任务微调与评估
在开始训练之前，最好在huggingface上手动下载一个bert-base-uncased的模型权重和配置文件，因为如果自动下载的话，可能出现网络问题，下载地址
必要的文件是：

pytroch_model.bin
tokenizer.json
tokenizer_config.json
config.json
vocab.txt
在你的代码中需要改一下路径
from transformers import BertModel

model = BertModel.from_pretrained("/path/to/local/bert-base-uncased")
下载官方的预训练权重（这些在官方的Readme里能看到）
14M
4M
开始训练（以Flickr30k为例子）

python Retrieval.py \
  --config ./configs/Retrieval_flickr.yaml \
  --output_dir output/Retrieval_flickr \
  --textencoder ./pretrained/bert-base-uncased \
  --checkpoint ./pretrained/ALBEF_14M.pth \
  --device cuda \
  --distributed False
这里如果多卡训练，distributed请设置为True
评估（以Flickr30k为例子）

python Retrieval.py \
  --config ./configs/Retrieval_flickr.yaml \
  --output_dir output/Retrieval_flickr \
  --textencoder ./pretrained/bert-base-uncased \
  --checkpoint output/Retrieval_flickr/checkpoint_best.pth \
  --evaluate
对比结果
ALBEF的官方论文地址
具体指标详见论文

可视化内容不再做赘述

其他问题解决
如遇到GPU显存不足，请自行降低图片分辨率或是降低batchsize的大小（在config的文件夹里）
Markdown 3034 字符2668 字数126 行数第 116 行, 第 8 列HTML 1961 字数100 段落
