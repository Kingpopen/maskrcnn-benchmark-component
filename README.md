# maskrcnn_benchmark with component branch
这是基于[maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)模型添加了多分支之后的模型介绍文件。

- [x] 在配置文件中添加了多分支相关类别的配置，可以指定是否使用多分支模型，简单实现了低耦合。
- [x] 添加了多分支数据集的相关数据集类
- [ ] 对预测的结果保存为labelme的格式



## 前期准备
### 1. requestment.txt
因为模型是基于maskrcnn_benchmark代码更改的，所以相关环境配置，
参照maskrcnn_benchmark的[INSTALL.md](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md)文件就好了。

### 2. 数据集格式
使用的数据集为COCO数据集格式，因为是多分支，所以多添加了一个分支的类别。
（如果你使用的数据集格式和我的不太一样，你可能需要修改成一样的格式才能正常使用:smile:）  
以下是数据集格式介绍：  
dataset_path  
├── train  
│   &ensp;├── annotations （包含训练集的标注文件)  
│   &ensp; &ensp; &ensp; &ensp;&ensp;└──instances_train2017.json  
│   &ensp;└── train2017  （包含训练集的图片）    
│   &ensp; &ensp; &ensp; &ensp;&ensp;├──1562433551586-2.jpg    
│   &ensp; &ensp; &ensp; &ensp;&ensp;├──1562433551587-2.jpg    
│   &ensp; &ensp; &ensp; &ensp;&ensp;├──....jpg  
│   &ensp; &ensp; &ensp; &ensp;&ensp;└──....jpg  
├── val  
│   &ensp;├── annotations  
│   &ensp; &ensp; &ensp; &ensp;&ensp;└──instances_val2017.json  
│   &ensp;└── val2017  
│   &ensp; &ensp; &ensp; &ensp;&ensp;├──1562433551576-2.jpg  
│   &ensp; &ensp; &ensp; &ensp;&ensp;├──1562433551577-2.jpg  
│   &ensp; &ensp; &ensp; &ensp;&ensp;├──....jpg  
│   &ensp; &ensp; &ensp; &ensp;&ensp;└──....jpg  
└── test  
&ensp;&ensp;&ensp;&ensp;├── annotations  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;└──instances_test2017.json  
&ensp;&ensp;&ensp;&ensp;└── test2017  
&ensp; &ensp; &ensp; &ensp; &ensp;&ensp;├──1562433551586-2.jpg  
&ensp; &ensp; &ensp; &ensp; &ensp;&ensp;├──1562433551586-2.jpg  
&ensp; &ensp; &ensp; &ensp; &ensp;&ensp;├──....jpg  
&ensp; &ensp; &ensp; &ensp; &ensp;&ensp;└──....jpg  

instances_test2017.json文件格式介绍：
(**如果你的格式和下面不一致，需要改成和下面一致的格式:smile:**)：
```python {.line-numbers}
{
        “images”:[
        {
            "height":...
            "width":...
            "id":...
            "filename_name":...(纯图片名 不带路径)
            }
        ...
        ...
        ],

        “categories”:[ (存放损伤标签)
        {
            "supercategory": "scratch",
            "id": 1,
            "name": "scratch"
            }
        ...
        ...
        ],

        "components": [(存放零件标签)
        {
            "supercategory": "bumper",
            "id": 1,
            "name": "bumper"
        }
        ...
        ...
        ],

        "annotations":[
        {
            "iscrowd": 0,
            "image_id": 1,
            "bbox": [...],
            "segmentation": [
                [[...]]
            ],
            "category_id": 1,
            "component_id": 1,
            "id": 1,
            "area": 18610.059492
        },
        ...
        ...
        ],  
    }
```





## 文件介绍
主要涉及**your_project_path/maskrcnn_benchmark/data**中的相关文件。
### 1. 数据集处理文件
因为本项目添加了多分支的模型，所以添加了一个处理多分支数据集的类，具体见 ：
**your_project_path/maskrcnn_benchmark/data/datasets/cardamage_util** (这是处理多分支COCO数据标注文件导入的工具包)
**your_project_path/maskrcnn_benchmark/data/datasets/cardamage.py** (这是生成pytorch的Dataset类型的类)


### 2. Box_head文件
项目的多分支主要添加在**your_project_path/maskrcnn_benchmark/modeling/roi_heads/box_head**当中：
#### 2.1 添加ROIBoxHead_component类
这个类添加在 **your_project_path/maskrcnn_benchmark/modeling/roi_heads/box_head/box_head.py**文件中
```python {.line-numbers}
# 包含有零件分支的ROIBoxHead
class ROIBoxHead_component(torch.nn.Module):
    """
    Generic Box Head class.
    ROIBoxHead的主要内容是：1、特征提取  feature extractor
                         2、边框和类别预测
                         3、后续处理（测试阶段的非极大值抑制）

    """

    def __init__(self, cfg, in_channels):
        ...
        ...

    def forward(self, features, proposals, targets=None):
        ...
        ...
```

#### 2.2 添加FPNPredictor_component类
这个类添加在 **your_project_path/maskrcnn_benchmark/modeling/roi_heads/box_head/roi_box_predictors.py**文件中

```python {.line-numbers}
# 在注册器中注册该类，方便在配置文件中添加该模块
@registry.ROI_BOX_PREDICTOR.register("FPNPredictor_component")
class FPNPredictor_component(nn.Module):
    def __init__(self, cfg, in_channels):
        ...
        ...
    def forward(self, x):
        ...
        ...
```
#### 2.3 添加FastRCNNLossComputation_component类
这个类添加在 
**your_project_path/maskrcnn_benchmark/modeling/roi_heads/box_head/loss.py**文件中,主要为了把零件的loss添加到整体的loss当中去。
```python {.line-numbers}
class FastRCNNLossComputation_component(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    对Faster-RCNN部分的loss进行计算
    """

    def __init__(
        self,
        proposal_matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg=False
    ):
        ...
        ...

    def match_targets_to_proposals(self, proposal, target):
        ...
        ...

    # 计算出所有预测边框所对应的GT边框
    def prepare_targets(self, proposals, targets):
        ...
        ...

    def subsample(self, proposals, targets):
        ...
        ...

    # 添加component类的损失计算
    def __call__(self, class_logits, component_logits, box_regression):
        ...
        ...
```

#### 2.4 添加PostProcessor_component类
这个类添加在
**your_project_path/maskrcnn_benchmark/modeling/roi_heads/box_head/inference.py**,主要用于inference过程，针对一些instances的筛选操作。
```python {.line-numbers}
# 添加了零件分支的ROI 后处理类
class PostProcessor_component(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results

    从一系列的类别分类得分，边框回归以及proposals中，计算post-processed boxes,
    以及应用NMS得到最后的结果。
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
        bbox_aug_enabled=False
    ):
        ...
        ...

    def forward(self, x, boxes):
        ...
        ...

    def prepare_boxlist(self, boxes, scores, component_scores, image_shape):
        ...
        ...

    def filter_results(self, boxlist, num_classes, num_components):
        ...
        ...
```

### 3. 配置文件
主要涉及**your_project_path/maskrcnn_benchmark/config/**中的相关文件。
#### 3.1 添加COMPONENT的配置
在**your_project_path/maskrcnn_benchmark/config/defaults.py**中添加COMPONENT的配置：
```python {line-numbers}
# 在ROI HEADS options的对应区域添加 component_branch变量，以及num_components
# 这样就可以在配置文件中设置是否使用 component_branch分支了
_C.MODEL.ROI_HEADS.COMPONENT_BRANCH = False

# 这样就可以在配置文件中设置num_components的数目了
_C.MODEL.ROI_BOX_HEAD.NUM_COMPONENTS = 33
```

#### 3.2 添加数据集路径配置
在**your_project_path/maskrcnn_benchmark/config/paths_catalog.py**中添加多分支数据集路径配置：
```python {.line-numbers}
#在 DatasetCatalog类中修改DATA_DIR路径,以及在DATASETS中添加多分支数据集信息
class DatasetCatalog(object):
    DATA_DIR = "your_dataset_root_path"
    DATASETS = {
        # 添加car damage 数据集路径
        "cardamage_2017_train": {
            "img_dir": "train/train2017",
            "ann_file": "train/annotations/instances_train2017.json"
        },
        "cardamage_2017_val": {
            "img_dir": "val/val2017",
            "ann_file": "val/annotations/instances_val2017.json"
        },
    ......
```

#### 3.3 更改配置文件
在**your_project_path/configs/cardamage/e2e_mask_rcnn_R_50_FPN_1x_ali.yaml**中更改模型相应参数配置：
```python {line-numbers}
  #在ROI_HEADS区域设置零件多分支
  ROI_HEADS:
    # 添加零件多分支
    COMPONENT_BRANCH: True

  # 将ROI_BOX_HEAD设置为:"FPNPredictor_component"
  ROI_BOX_HEAD:
    NUM_CLASSES: 5
    # 添加零件类别数（背景也包含在内）
    NUM_COMPONENTS: 7
    PREDICTOR: "FPNPredictor_component"

  # 将数据集设置成多分支的数据集
  DATASETS:
    TRAIN: ("cardamage_2017_train",)
    TEST: ("cardamage_2017_val",)
```

## 模型运行
模型运行和maskrcnn_benchmark基本一致:smile:
### 1. train
```python
python ./tools/train_net.py  --config-file  your_config_file
```

### 2. inference
```python 
# 后续的指标计算，请使用cardamage_evaluation_api计算
python ./tools/test_net.py  --config-file  your_config_file
```
### 3. predict
后续生成labelme格式数据的操作还未完善好:disappointed_relieved:
```python
python  ./demo/cardamage/predict.py
```
