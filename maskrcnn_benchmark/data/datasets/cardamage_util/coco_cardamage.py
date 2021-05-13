__author__ = 'tylin'
__version__ = '2.0'
# Interface for accessing the Microsoft COCO dataset.
#这是一个接口去处理Microsoft的COCO数据集  （现在我将它更改为处理我自己的数据集）


# Microsoft COCO is a large image dataset designed for object detection,
# segmentation, and caption generation. pycocotools is a Python API that
# assists in loading, parsing and visualizing the annotations in COCO.
# Please visit http://mscoco.org/ for more information on COCO, including
# for the data, paper, and tutorials. The exact format of the annotations
# is also described on the COCO website. For example usage of the pycocotools
# please see pycocotools_demo.ipynb. In addition to this API, please download both
# the COCO images and annotations in order to run the demo.
'''
Microsoft COCO 是一个大型的用于目标检测、分割以及caption generation的图像数据集,pycocotools是一个Python API
来协助导入、解析和可视化COCO中的annotation是，请访问http://mscoco.org/查看更多关于COCO的信息，其中包含有数据、论文以及教程。
annotations的格式也详细的写在COCO网址上。如果想要看pycocotools的使用说明，可以查看pycocotools_demo.ipynb文件，除了这个API，
如果想要运行demo，请你还要下载相应的COCO图片和annotations文件。
'''

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).
'''
使用API的另一种方式是将annotations导入python的字典中
使用API提供额外的有用的函数。注意这个API支持‘instance’和‘caption’标注。
如果是caption，并不是所有的函数都被定义了。（例如’类别‘就没有被定义）
'''
# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  annToMask  - Convert segmentation in an annotation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".
'''
以下的API函数都是被定义的：
COCO           - COCO API的类，它主要用来导入COCO的标注文件以及准备数据结构
decodeMask     - 解码二进制mask M（这个我不太明白） 不太清楚encode和decode得到的结果是个什么
encodeMask     - 编码二进制mask M（这个我不太明白）
getAnnIds      - 返回满足过滤条件的ann的ids
getCatIds      - 返回满足过滤条件的类别的ids
getImgIds      - 返回满足过滤条件的图片的ids
loadAnns       - 通过指定相应的ids导入anns
loadCats       - 通过指定相应的ids导入类别
loadImgs       - 通过指定相应的ids导入图片
annToMask      - 将annotation中的segmentation转化为二进制的mask
showAnns       - 展示指定的annotation
loadRes        - 加载算法结果并创建用于访问它们的API（这个地方我不是很懂干啥用的）
download       - 从mscoco.org服务器下载COCO 图片

API中’ann‘ = annotation  'cat' = category  'img' = image

'''
# See also COCO>decodeMask,
# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>annToMask, COCO>showAnns

# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import json
import time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import copy
import itertools
from pycocotools import mask as maskUtils
from pycocotools import coco
import os
from collections import defaultdict
import sys
# 根据不同的python版本导入不同的包
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

# 我自己写的COCOCarDamage类
class COCOCarDamage:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file（标注文件的地址）
        :param image_folder (str): location to the folder that hosts images.（存放图片的文件夹地址）
        :return:
        """
        # load dataset   comps是指零件类别
        self.dataset, self.anns, self.cats, self.comps, self.imgs = dict(), dict(), dict(), dict(), dict()
        # 字典的value是一个list   添加了self.compToImgs成员
        self.imgToAnns, self.catToImgs , self.compToImgs = defaultdict(list), defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    # 给类变量初始化生成索引（重要）
    def createIndex(self):
        # create index
        print('creating index...')
        # imgs中key为img的id值，value为该id值所对应的img信息
        anns, cats, comps, imgs = {}, {}, {}, {}
        # imgToAnns中key为img的id，value是id所对应的anns的list
        imgToAnns, catToImgs, compToImgs = defaultdict(list), defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        # 添加对components的操作
        if 'components' in self.dataset:
            for comp in self.dataset['components']:
                comps[comp['id']] = comp


        if 'annotations' in self.dataset and 'categories' in self.dataset and 'components':
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])
                compToImgs[ann['component_id']].append(ann['image_id'])
        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.compToImgs = compToImgs
        self.imgs = imgs
        self.cats = cats
        self.comps = comps

    # 打印出annotation文件的信息（不太重要）
    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))

    # 获得满足过滤条件的anns对应的ids（例如：通过imgid来查找anns的id，通过catId来查找anns的id）
    def getAnnIds(self, imgIds=[], catIds=[], compIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               compIds  (int array)     : get anns for given comps
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        # 判断imgIds和catIds是否为list，如果不是就把它变成list
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]
        # 添加compIds
        compIds = compIds if _isArrayLike(compIds) else [compIds]

        if len(imgIds) == len(catIds) == len(compIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            # 首先通过imgIds进行筛选
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            # 接着通过catIds进行筛选
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann['category_id'] in catIds]

            # 接着通过compsIds进行筛选
            anns = anns if len(compIds) == 0 else [ann for ann in anns if ann['component_id'] in compIds]

            # 最后通过area值进行筛选
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    # 获得满足过滤条件的categories对应的ids（按照参数从左向右的顺序优先级依次递减）
    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names（通过类别名进行过滤）
        :param supNms (str array)  : get cats for given supercategory names（通过supercategory进行过滤）
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    # 获得满足过滤条件的components对应的ids（按照参数从左向右的顺序优先级依次递减）
    def getCompIds(self, compNms=[], supNms=[], compIds=[]):
        """
        filtering parameters. default skips that filter.
        :param compNms (str array)  : get cats for given comp names（通过零件的类别名进行过滤）
        :param supNms (str array)  : get cats for given supercategory names（通过supercategory进行过滤）
        :param compIds (int array)  : get cats for given comp ids
        :return: ids (int array)   : integer array of comp ids
        """
        compNms = compNms if _isArrayLike(compNms) else [compNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        compIds = compIds if _isArrayLike(compIds) else [compIds]

        if len(compNms) == len(supNms) == len(compIds) == 0:
            comps = self.dataset['components']
        else:
            comps = self.dataset['components']
            comps = comps if len(compNms) == 0 else [comp for comp in comps if comp['name']          in compNms]
            comps = comps if len(supNms) == 0 else [comp for comp in comps if comp['supercategory'] in supNms]
            comps = comps if len(compIds) == 0 else [comp for comp in comps if comp['id']            in compIds]
        ids = [comp['id'] for comp in comps]
        return ids

    #获得满足过滤条件的img对应的ids(添加了compIds)
    def getImgIds(self, imgIds=[], catIds=[], compIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids（在imgIds这个范围内选择）
        :param catIds (int array) : get imgs with all given cats（在catIds这个范围内进行选择）
        :param compIds (int array) : get imgs with all given comps（在compIds这个范围内进行选择）
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]
        compIds = compIds if _isArrayLike(compIds) else [compIds]


        if len(imgIds) == len(catIds) == len(compIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        #    添加零件的判断
            for i, compId in enumerate(compIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.compToImgs[compId])
                else:
                    ids &= set(self.compToImgs[compId])

        return list(ids)

    # 通过annotation的ids获取anns
    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    # 通过categories的ids获取对应的类别名信息
    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    # 通过components的ids获取对应的类别名信息
    def loadComps(self, ids=[]):
        """
        Load comps with the specified ids.
        :param ids (int array)       : integer ids specifying dams
        :return: comps (object array) : loaded comp objects
        """
        if _isArrayLike(ids):
            return [self.comps[id] for id in ids]
        elif type(ids) == int:
            return [self.comps[ids]]

    # 通过对应的img的ids获取对应的图片信息
    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    # 对annotation进行可视化处理
    def showAnns(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        # 判断数据集的类型 （instance 、 caption ）
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')
        # 如果是进行分割任务
        if datasetType == 'instances':
            ax = plt.gca()
            ax.set_autoscale_on(False)

            polygons = []
            color = []
            for ann in anns:
                c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg)/2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    else:
                        # mask
                        t = self.imgs[ann['image_id']]
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                        else:
                            rle = [ann['segmentation']]

                        # print("the shape of segmentation:", np.array(ann["segmentation"]).shape)

                        m = maskUtils.decode(rle)
                        # print("the type of m:", type(m))
                        # print("the shape of m:", m.shape)

                        img = np.ones( (m.shape[0], m.shape[1], 3) )
                        if ann['iscrowd'] == 1:
                            color_mask = np.array([2.0,166.0,101.0])/255
                        if ann['iscrowd'] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:,:,i] = color_mask[i]
                        ax.imshow(np.dstack( (img, m*0.5) ))
                if 'keypoints' in ann and type(ann['keypoints']) == list:
                    # turn skeleton into zero-based index
                    sks = np.array(self.loadCats(ann['category_id'])[0]['skeleton'])-1
                    kp = np.array(ann['keypoints'])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if np.all(v[sk]>0):
                            plt.plot(x[sk],y[sk], linewidth=3, color=c)
                    plt.plot(x[v>0], y[v>0],'o',markersize=8, markerfacecolor=c, markeredgecolor='k',markeredgewidth=2)
                    plt.plot(x[v>1], y[v>1],'o',markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)

            # print("the shape of polygons:", np.array(polygons).shape)

            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)

            # plt.savefig("./save.jpg")
            # plt.show()
            #
            # fig = plt.figure()
            # ax = fig.add_subplot(1, 1, 1)
            #
            # rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3)
            # circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.3)
            # pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]],
            #                    color='g', alpha=0.5)
            #
            # ax.add_patch(rect)
            # ax.add_patch(circ)
            # ax.add_patch(pgon)
            # plt.savefig("./wuwuuw.jpg")


        elif datasetType == 'captions':
            for ann in anns:
                print(ann['caption'])


    # 生成预测结果的COCOCardamage对象的处理(通过传入模型预测结果的annotation，生成一个COCOCardamage对象)
    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCOCarDamage()
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()

        # Check result type in a way compatible with Python 2 and 3.
        if PYTHON_VERSION == 2:
            is_string = isinstance(resFile, np.basestring)  # Python 2
        elif PYTHON_VERSION == 3:
            is_string = isinstance(resFile, str)  # Python 3
        if is_string:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
               'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id+1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            # 添加component类别
            res.dataset['components'] = copy.deepcopy(self.dataset['components'])

            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2]*bb[3]
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            # 添加components类别
            res.dataset['components'] = copy.deepcopy(self.dataset['components'])

            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'keypoints' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            # 添加components类别
            res.dataset['components'] = copy.deepcopy(self.dataset['components'])

            for id, ann in enumerate(anns):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['area'] = (x1-x0)*(y1-y0)
                ann['id'] = id + 1
                ann['bbox'] = [x0,y0,x1-x0,y1-y0]
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res


    def loadNumpyAnnotations(self, data):
        """
        Convert result data from a numpy array [Nx8] where each row contains {imageID,x1,y1,w,h,score,class,component}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        print('Converting ndarray to lists...')
        assert(type(data) == np.ndarray)
        print(data.shape)
        # assert(data.shape[1] == 7)
        assert (data.shape[1] == 8)  #多加了一个损伤类别
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                print('{}/{}'.format(i, N))
            ann += [{
                'image_id'  : int(data[i, 0]),
                'bbox'  : [ data[i, 1], data[i, 2], data[i, 3], data[i, 4] ],
                'score' : data[i, 5],
                'category_id': int(data[i, 6]),
                'component_id': int(data[i, 7])
                }]
        return ann

    def annToRLE(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """

        t = self.imgs[ann['image_id']]
        h, w = t['height'], t['width']
        segm = ann['segmentation']
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    # 将annotation转化为mask
    def annToMask(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann)
        m = maskUtils.decode(rle)
        return m

    # 类别转换（用于评估阶段，需要将多分支的输出类别整合成单个类别）输入COCOCarDamage对象 返回 COCO对象
    '''
    我的coco格式标注数据格式介绍：
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
    '''
    # 所以将多分支类别改为单分支类别 主要是更改categories 以及 annotation中的category_id
    # 6种零件4种损伤 转化为单一的24种类别
    def transformTo24(self):
        '''
        Args:
            cococardamage: COCOCarDamage对象
        Returns:
            COCO对象
        '''
        # 生成24类别名 以及 其id
        damage_name = ["scratch", "indentation", "crack", "perforation"]
        component_name = ["bumper", "fender", "light", "rearview", "windshield", "others"]
        categories_24_list = []
        category_24_id = 0
        for component in component_name:
            for damage in damage_name:
                category_dict = {}
                category_24_id += 1
                category_24_label = component + " " + damage
                category_dict["supercategory"] = category_24_label
                category_dict["id"] = category_24_id
                category_dict["name"] = category_24_label
                categories_24_list.append(category_dict)

        # 获取已经读取的标注文件数据，相当于instance.json文件中的内容
        dataset = self.dataset
        dataset["categories"] = categories_24_list
        # 对annotation中类别进行修改
        annotations = dataset["annotations"]
        new_annotations = []
        for ann in annotations:
            category_id = ann["category_id"]
            component_id = ann["component_id"]
            new_category_id = (int(component_id) - 1) * len(damage_name) + int(category_id)
            ann["category_id"] = new_category_id
            new_annotations.append(ann)
        dataset["annotations"] = new_annotations
        Coco = coco.COCO()
        Coco.dataset = dataset
        Coco.createIndex()
        return Coco

    # 13种零件4种损伤 转化为单一的52种类别
    def transformTo52(self):
        '''
               Args:
                   cococardamage: COCOCarDamage对象
               Returns:
                   COCO对象
               '''
        # 生成52类别名 以及 其id
        damage_name = ["scratch", "indentation", "crack", "perforation"]
        component_name = ["front bumper", "rear bumper", "front fender",
                       "rear fender", "door", "rear taillight",
                       "headlight", "hood", "luggage cover",
                       "radiator grille", "bottom side", "rearview mirror",
                       "license plate"]

        categories_52_list = []
        category_52_id = 0
        for component in component_name:
            for damage in damage_name:
                category_dict = {}
                category_52_id += 1
                category_52_label = component + " " + damage
                category_dict["supercategory"] = category_52_label
                category_dict["id"] = category_52_id
                category_dict["name"] = category_52_label
                categories_52_list.append(category_dict)

        # 获取已经读取的标注文件数据，相当于instance.json文件中的内容
        dataset = self.dataset
        dataset["categories"] = categories_52_list
        # 对annotation中类别进行修改
        annotations = dataset["annotations"]
        new_annotations = []
        for ann in annotations:
            category_id = ann["category_id"]
            component_id = ann["component_id"]
            new_category_id = (int(component_id) - 1) * len(damage_name) + int(category_id)
            ann["category_id"] = new_category_id
            new_annotations.append(ann)
        dataset["annotations"] = new_annotations
        Coco = coco.COCO()
        Coco.dataset = dataset
        Coco.createIndex()
        return Coco

    # 32种零件5种损伤 转化为单一的160种类别
    def transformTo160(self):
        # 生成52类别名 以及 其id
        damage_name = ["scratch", "indentation", "crack", "perforation"]
        component_name = [
            "front bumper", "rear bumper",
            "front bumper grille", "front windshield",
            "rear windshield", "front tire",
            "rear tire", "front side glass",
            "rear side glass", "front fender",
            "rear fender", "front mudguard",
            "rear mudguard", "turn signal",
            "front door", "rear door",
            "rear outer taillight", "rear inner taillight",
            "headlight", "fog light",
            "hood", "luggage cover",
            "roof", "steel ring",
            "radiator grille", "a pillar",
            "b pillar", "c pillar",
            "d pillar", "bottom side",
            "rearview mirror", "license plate"]

        categories_160_list = []
        category_160_id = 0
        for component in component_name:
            for damage in damage_name:
                category_dict = {}
                category_160_id += 1
                category_160_label = component + " " + damage
                category_dict["supercategory"] = category_160_label
                category_dict["id"] = category_160_id
                category_dict["name"] = category_160_label
                categories_160_list.append(category_dict)

        # 获取已经读取的标注文件数据，相当于instance.json文件中的内容
        dataset = self.dataset
        dataset["categories"] = categories_160_list
        # 对annotation中类别进行修改
        annotations = dataset["annotations"]
        new_annotations = []
        for ann in annotations:
            category_id = ann["category_id"]
            component_id = ann["component_id"]
            new_category_id = (int(component_id) - 1) * len(damage_name) + int(category_id)
            ann["category_id"] = new_category_id
            new_annotations.append(ann)
        dataset["annotations"] = new_annotations
        Coco = coco.COCO()
        Coco.dataset = dataset
        Coco.createIndex()
        return Coco


if __name__ == '__main__':
    json_file = "/home/pengjinbo/kingpopen/Dataset/Car/damage/coco/ali_dataset_multi/val/annotations/instances_val2017.json"
    cardamage = COCOCarDamage(json_file)
    Coco = cardamage.transformTo24()
    anns = Coco.loadAnns(Coco.getAnnIds([1]))

    print("image id:", anns[0]["image_id"])
    print("len of anns:", len(anns))
    print("anns:", anns)

    for ann in anns:
        print("category_id:", ann["category_id"])

    print("categories:", Coco.cats)




