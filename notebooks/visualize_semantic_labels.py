#%%
import os
from re import L
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import namedtuple
# from IPython.display import display, clear_output
from time import sleep
import cv2
import glob
import imageio
# %%
CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                    'has_instances', 'ignore_in_eval', 'color'])

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
        CityscapesClass('wall',                 12, 2, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 2, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 13, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 13, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 13, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 2, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 13, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 3, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 4, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 5, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 5, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 6, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 7, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 8, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 9, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 9, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 10, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 13, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 13, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 11, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 12, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('unknown',              255, 13, 'void', 7, True, False, (0, 0, 0)),
        CityscapesClass('license plate',        -1, 13, 'vehicle', 7, False, True, (0, 0, 0)),
    ]

cs = [
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
        CityscapesClass('license plate',        -1, 19, 'vehicle', 7, False, True, (0, 0, 0)),
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
        CityscapesClass('unknown',              0, 19, 'vehicle', 7, False, False, (0, 0, 0)),
    ]

n_classes = 16
valid_classes = [3,4,2,21,5,7,15,9,6,1,10,17,8,19,12,11,]
class_map = dict(zip(valid_classes, range(n_classes)))


# %%

def convert(semantic_map, mapping):
    # id_to_trainId = {cs_class.id: cs_class.train_id for cs_class in mapping}
    mask_copy = np.ones_like(semantic_map)*n_classes
    for k, v in mapping.items():
        mask_copy[semantic_map == k] = v
    return mask_copy

palette = []
colors = {cs_class.train_id: cs_class.color for cs_class in synthia_full}
for train_id, color in sorted(colors.items(), key=lambda item: item[0]):
    R, G, B = color
    palette.extend((R, G, B))

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)
# %%



# %%
for i, label_path in enumerate(sorted(glob.glob("/media/data4/SYNTHIA_RAND_CITYSCAPES/GT/LABELS/0002005.png"))):
    # print(label_path)
    # label = Image.open(label_path)
    label = np.asarray(imageio.imread(label_path, format='PNG-FI'))[:,:,0]
    # image = Image.open(label_path.replace("step1_dacs_da_semantic_encoded", "step1_dacs_da"))
    # label = np.asarray(imageio.imread(label_path, format='PNG-FI'))[:,:,0]
    np_label = np.array(label)

    print(np_label.max())
    print(np.unique(np_label))

    np_label = convert(np_label, class_map)
    print(np_label.max(), np_label.min())
    print(np.unique(np_label))
    # np_label[np_label==255]=19
    label = Image.fromarray(np_label)

    label = label.convert('P')
    label.putpalette(palette)

    display(label)
    # display(image)
    # label.save(f"/data/aCardace/iterativive_da/qualitatives/predictions_from_gta_baseline/baseline_{i}.png")
    # sleep(4)
    # clear_output()
    break

#%%

# import glob
# fout = open("/data/aCardace/iterativive_da/splits/synscapes/final.txt", "w")
# # for i, image_path in enumerate(glob.glob("/data/aCardace/datasets/Synscapes/img/final_step/*")):
# for i in range(2975):
#     # image_name = image_path.split("/")[-1]
#     # print(image_name)

#     # image_path = image_path.replace("/data/aCardace/datasets/", "")
#     image_path = f"Synscapes/img/final_step/{i}.png"
#     label_path = image_path.replace("final_step", "final_step_semantic_encoded")
#     line = f"{image_path};{label_path};{image_path}\n"
#     fout.write(line)
# print(i)
# fout.close()
# %%
