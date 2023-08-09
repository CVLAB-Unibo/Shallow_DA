from torch.utils.data import Dataset
from collections import namedtuple
import os
import torch
import numpy as np
import PIL
from PIL import Image, ImageFile
import hydra
import cv2
import imageio

# from torchvision.io import read_image
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SegmentationDataset(Dataset):

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                    'has_instances', 'ignore_in_eval', 'color'])
    
    cs = [
        CityscapesClass('road',                 0,  0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             1,  1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('building',             2,  2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 3,  3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                4,  4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('pole',                 5,  5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('traffic light',        6,  6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         7,  7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           8,  8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              9,  9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  10, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               11, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                12, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  13, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                14, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  15, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('train',                16, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           17, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              18, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('void',                 19, 19, 'vehicle', 7, False, False, (0, 0, 0)),
    ]

    cs_original = [
        CityscapesClass('unlabeled',            0, 19, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 19, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 19, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 19, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 19, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 19, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 19, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 19, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 19, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 19, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 19, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 19, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 19, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 19, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 19, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('unknown',              255, 19, 'void', 7, True, False, (0, 0, 0)),
        CityscapesClass('license plate',        -1, 19, 'vehicle', 7, False, True, (255, 255, 255)),
    ]

    cs_nthu = [
        CityscapesClass('road',                 0,  0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             1,  1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('building',             2,  2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 3,  13, 'construction', 2, False, True, (102, 102, 156)),
        CityscapesClass('fence',                4,  13, 'construction', 2, False, True, (190, 153, 153)),
        CityscapesClass('pole',                 5,  13, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        6,  3, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         7,  4, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           8,  5, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              9,  5, 'nature', 4, False, True, (107, 142, 35)),
        CityscapesClass('sky',                  10, 6, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               11, 7, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                12, 8, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  13, 9, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                14, 9, 'vehicle', 7, True, True, (0, 0, 142)),
        CityscapesClass('bus',                  15, 10, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('train',                16, 13, 'vehicle', 7, True, True, (0, 80, 100)),
        CityscapesClass('motorcycle',           17, 11, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              18, 12, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('void',                 19, 13, 'vehicle', 7, False, False, (0, 0, 0)),
    ]

    nthu_original = [
        CityscapesClass('road',                 0,  0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             1,  1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('building',             2,  2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('traffic light',        3,  3, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         4,  4, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           5,  5, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('sky',                  6,  6, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               7,  7, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                8,  8, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  9,  9, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('bus',                  10, 10, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('motorcycle',           11, 11, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              12, 12, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('void',                 13, 13, 'vehicle', 7, False, False, (0, 0, 0)),
    ]

    nthu = [
            CityscapesClass('unlabeled',            0, 13, 'void', 0, False, True, (0, 0, 0)),
            CityscapesClass('ego vehicle',          1, 13, 'void', 0, False, True, (0, 0, 0)),
            CityscapesClass('rectification border', 2, 13, 'void', 0, False, True, (0, 0, 0)),
            CityscapesClass('out of roi',           3, 13, 'void', 0, False, True, (0, 0, 0)),
            CityscapesClass('static',               4, 13, 'void', 0, False, True, (0, 0, 0)),
            CityscapesClass('dynamic',              5, 13, 'void', 0, False, True, (111, 74, 0)),
            CityscapesClass('ground',               6, 13, 'void', 0, False, True, (81, 0, 81)),
            CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
            CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
            CityscapesClass('parking',              9, 13, 'flat', 1, False, True, (250, 170, 160)),
            CityscapesClass('rail track',           10, 13, 'flat', 1, False, True, (230, 150, 140)),
            CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
            CityscapesClass('wall',                 12, 2, 'construction', 2, False, False, (70, 70, 70)),
            CityscapesClass('fence',                13, 2, 'construction', 2, False, False, (70, 70, 70)),
            CityscapesClass('guard rail',           14, 13, 'construction', 2, False, True, (180, 165, 180)),
            CityscapesClass('bridge',               15, 13, 'construction', 2, False, True, (150, 100, 100)),
            CityscapesClass('tunnel',               16, 13, 'construction', 2, False, True, (150, 120, 90)),
            CityscapesClass('pole',                 17, 2, 'object', 3, False, False, (70, 70, 70)),
            CityscapesClass('polegroup',            18, 13, 'object', 3, False, True, (153, 153, 153)),
            CityscapesClass('traffic light',        19, 3, 'object', 3, False, False, (250, 170, 30)),
            CityscapesClass('traffic sign',         20, 4, 'object', 3, False, False, (220, 220, 0)),
            CityscapesClass('vegetation',           21, 5, 'nature', 4, False, False, (107, 142, 35)),
            CityscapesClass('terrain',              22, 5, 'nature', 4, False, False, (107, 142, 35)),
            CityscapesClass('sky',                  23, 6, 'sky', 5, False, False, (70, 130, 180)),
            CityscapesClass('person',               24, 7, 'human', 6, True, False, (220, 20, 60)),
            CityscapesClass('rider',                25, 8, 'human', 6, True, False, (255, 0, 0)),
            CityscapesClass('car',                  26, 9, 'vehicle', 7, True, False, (0, 0, 142)),
            CityscapesClass('truck',                27, 9, 'vehicle', 7, True, False, (0, 0, 142)),
            CityscapesClass('bus',                  28, 10, 'vehicle', 7, True, False, (0, 60, 100)),
            CityscapesClass('caravan',              29, 13, 'vehicle', 7, True, True, (0, 0, 90)),
            CityscapesClass('trailer',              30, 13, 'vehicle', 7, True, True, (0, 0, 110)),
            CityscapesClass('train',                31, 13, 'vehicle', 7, True, False, (0, 80, 100)),
            CityscapesClass('motorcycle',           32, 11, 'vehicle', 7, True, False, (0, 0, 230)),
            CityscapesClass('bicycle',              33, 12, 'vehicle', 7, True, False, (119, 11, 32)),
            CityscapesClass('unknown',              255, 13, 'void', 7, True, False, (0, 0, 0)),
            CityscapesClass('license plate',        -1, 13, 'vehicle', 7, False, True, (0, 0, 0)),
        ]

    synthia = [
            CityscapesClass('road',                 0,  0, 'flat', 1, False, False, (128, 64, 128)),
            CityscapesClass('sidewalk',             1,  1, 'flat', 1, False, False, (244, 35, 232)),
            CityscapesClass('building',             2,  2, 'construction', 2, False, False, (70, 70, 70)),
            CityscapesClass('wall',                 3,  3, 'construction', 2, False, False, (102, 102, 156)),
            CityscapesClass('fence',                4,  4, 'construction', 2, False, False, (190, 153, 153)),
            CityscapesClass('pole',                 5,  5, 'object', 3, False, False, (153, 153, 153)),
            CityscapesClass('traffic light',        6,  6, 'object', 3, False, False, (250, 170, 30)),
            CityscapesClass('traffic sign',         7,  7, 'object', 3, False, False, (220, 220, 0)),
            CityscapesClass('vegetation',           8,  8, 'nature', 4, False, False, (107, 142, 35)),
            CityscapesClass('sky',                  9,  9, 'sky', 5, False, False, (70, 130, 180)),
            CityscapesClass('person',               10, 10, 'human', 6, True, False, (220, 20, 60)),
            CityscapesClass('rider',                11, 11, 'human', 6, True, False, (255, 0, 0)),
            CityscapesClass('car',                  12, 12, 'vehicle', 7, True, False, (0, 0, 142)),
            CityscapesClass('bus',                  13, 13, 'vehicle', 7, True, False, (0, 60, 100)),
            CityscapesClass('motorcycle',           14, 14, 'vehicle', 7, True, False, (0, 0, 230)),
            CityscapesClass('bicycle',              15, 15, 'vehicle', 7, True, False, (119, 11, 32)),
            CityscapesClass('unknown',              16, 16, 'void', 7, True, False, (0, 0, 0)),
        ]


    synthia_full = [
        CityscapesClass('road',                 3, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             4, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('building',             2, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 21, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                5, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('pole',                 7, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('traffic light',        15, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         9, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           6, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              16, 16, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  1, 9, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               10, 10, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                17, 11, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  8, 12, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                18, 16, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  19, 13, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('train',                20, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           12, 14, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              11, 15, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('unknown',              0, 16, 'vehicle', 7, False, False, (0, 0, 0)),
    ]

    cs2synthia = [
            CityscapesClass('road',                  0, 0, 'flat', 1, False, False, (128, 64, 128)),
            CityscapesClass('sidewalk',              1, 1, 'flat', 1, False, False, (244, 35, 232)),
            CityscapesClass('building',              2, 2, 'construction', 2, False, False, (70, 70, 70)),
            CityscapesClass('wall',                  3, 3, 'construction', 2, False, False, (102, 102, 156)),
            CityscapesClass('fence',                 4, 4, 'construction', 2, False, False, (190, 153, 153)),
            CityscapesClass('pole',                  5, 5, 'object', 3, False, False, (153, 153, 153)),
            CityscapesClass('traffic light',         6, 6, 'object', 3, False, False, (250, 170, 30)),
            CityscapesClass('traffic sign',          7, 7, 'object', 3, False, False, (220, 220, 0)),
            CityscapesClass('vegetation',            8, 8, 'nature', 4, False, False, (107, 142, 35)),
            CityscapesClass('terrain',               9, 16, 'nature', 4, False, True, (152, 251, 152)),
            CityscapesClass('sky',                   10, 9, 'sky', 5, False, False, (70, 130, 180)),
            CityscapesClass('person',                11, 10, 'human', 6, True, False, (220, 20, 60)),
            CityscapesClass('rider',                 12, 11, 'human', 6, True, False, (255, 0, 0)),
            CityscapesClass('car',                   13, 12, 'vehicle', 7, True, False, (0, 0, 142)),
            CityscapesClass('truck',                 14, 16, 'vehicle', 7, True, True, (0, 0, 70)),
            CityscapesClass('bus',                   15, 13, 'vehicle', 7, True, False, (0, 60, 100)),
            CityscapesClass('train',                 16, 16,'vehicle', 7, True, True, (0, 80, 100)),
            CityscapesClass('motorcycle',            17, 14, 'vehicle', 7, True, False, (0, 0, 230)),
            CityscapesClass('bicycle',               18, 15, 'vehicle', 7, True, False, (119, 11, 32)),
            CityscapesClass('void',                  19, 16,'void', 7, True, False, (0, 0, 0)),
        ]

    def __init__(self, root, txt_file, mean=(0.485, 0.456, 0.406), std=(0.229, 0.225, 0.224), transform=None, encode=False, encoding=None, num_classes = 16):
    
        super().__init__()
        self.name = encoding
        self.num_classes = num_classes

        if encoding=="synthia_full":
            self.encoding = self.synthia
            # self.valid_classes = [3,4,2,21,5,7,15,9,6,1,10,17,8,19,12,11,]
            # self.id_to_trainId = dict(zip(self.valid_classes, range(self.num_classes)))
            # self.segmentationum_classes = ["unlabelled","Road","Sidewalk","Building","Wall",
            #     "Fence","Pole","Traffic_light","Traffic_sign","Vegetation",
            #     "sky","Pedestrian","Rider","Car","Bus",
            #     "Motorcycle","Bicycle",
            # ]
        
        elif encoding=="cs2synthia":
            self.encoding = self.cs2synthia
        elif encoding=="cs_original":
            self.encoding = self.cs_original
        elif encoding=="nthu":
            self.encoding = self.nthu
        elif encoding=="cs_nthu":
            self.encoding = self.cs_nthu
        elif encoding=="nthu_original":
            self.encoding = self.nthu_original
        else:
            self.encoding = self.cs

        # if not encoding=="synthia":
        self.id_to_trainId = {cs_class.id: cs_class.train_id for cs_class in self.encoding}
        self.segmentation_classes = [cs_class.name for cs_class in self.encoding if not cs_class.ignore_in_eval]
        self.palette = []
        self.files_txt = txt_file
        self.images = []
        self.labels = []
        self.root = root
        self.transform = transform
        self.encode = encode
        

        for line in open(os.path.join("/home/acardace/projects/shallow/", self.files_txt), 'r').readlines():
            splits = line.split(';')
            self.images.append(os.path.join(root, splits[0].strip()))
            self.labels.append(os.path.join(root, splits[1].strip()))

        self.colors = {cs_class.train_id: cs_class.color for cs_class in self.encoding}
        for train_id, color in sorted(self.colors.items(), key=lambda item: item[0]):
            R, G, B = color
            self.palette.extend((R, G, B))

        zero_pad = 256 * 3 - len(self.palette)
        for i in range(zero_pad):
            self.palette.append(0)

        self.mean = mean
        self.std = std

    def __getitem__(self, index):
        image_path, label_path = self.images[index], self.labels[index]

        image = Image.open(image_path).convert("RGB")
        if "synthia_full" == self.name:
            label = np.asarray(imageio.imread(label_path, format='PNG-FI'))[:,:,0]
        else:
            label = Image.open(label_path)
            
        if self.encode:
            label = np.array(label, dtype=np.uint8)
            label = self.convert(label, self.id_to_trainId)
            label = Image.fromarray(label)
       
        image, label = self.transform(image, label)
        edges = cv2.Canny(label.numpy().astype(np.uint8), 0, 10)
        edges = torch.tensor(edges).float()/255.0
        return image, label, edges.unsqueeze(0), label_path
    #     # return image, label, image_path, label_path


    # def convert(self, semantic_map, id_to_trainId):
    #     mask_copy = semantic_map.copy()
    #     for k, v in id_to_trainId.items():
    #         mask_copy[semantic_map == k] = v[]
    #     return mask_copy

    def convert(self, semantic_map, mapping):
    # id_to_trainId = {cs_class.id: cs_class.train_id for cs_class in mapping}
        mask_copy = np.ones_like(semantic_map)*self.num_classes
        for k, v in mapping.items():
            mask_copy[semantic_map == k] = v
        return mask_copy

    def colorize_mask(self, mask):
        mask = np.array(mask, dtype=np.uint8)
        mask = Image.fromarray(mask).convert("P")
        mask.putpalette(self.palette)
        return mask

    def re_normalize(self, image, gbr=False):
        image = np.array(image, np.float32)
        image = image.transpose((1, 2, 0))
        image *= self.std
        image += self.mean
        if gbr:
            return np.uint8(np.array(image)[:, :, ::-1])
        else:
            return np.uint8(image * 255)

    def __len__(self):
        return len(self.images)