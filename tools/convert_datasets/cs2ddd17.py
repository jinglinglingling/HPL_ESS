import cv2
import glob
import os
from tqdm import tqdm
import numpy as np

# Cityscapes
# Road	sidewalk	Bui. 	Wall 	Fence	Pole	Traffic light 	Traffic sign	Vegetation	  Terrain	  Sky    Person     rider    Car     truck    bus   train
#  0       1         2       3        4       5         6                 7              8           9         10       11        12      13       14      15     16

# DSEC
# Sky	Bui.	Fence 	Person	Pole	Road	Sidewalk	Veg.	Trans.	Wall	Traffic
#  0     1        2        3      4       5         6        7         8      9        10 

# DDD17
# flat  construction + sky  object  nature  human   vehicle
#  0             1             2       3      4        5

dsecid_to_ddd17id = {
    0: 1,
    1: 1,
    2: 1,
    3: 4,
    4: 2,
    5: 0,
    6: 0,
    7: 3,
    8: 5,
    9: 1,
    10: 2,
    255: 255
}

csid_to_ddd17id = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
    4: 1,
    5: 2,
    6: 2,
    7: 2,
    8: 3,
    9: 3,
    10: 1,
    11: 4,
    12: 4,
    13: 5,
    14: 5,
    15: 5,
    16: 5,
    17: 5,
    18: 5,
    255: 255
}

data_path = '/mnt/workspace/dingyiming/Codes/semi-supervised-uda/data/cityscapes/gtFine/test'
dir_list = sorted(glob.glob(data_path + '/*'))


for path in tqdm(dir_list):
    img_list = sorted(glob.glob(path + '/*_labelTrainIds.png'))
    for img_path in tqdm(img_list):
        label = cv2.imread(img_path)
        W, H, C = label.shape
        label_ddd17 = np.zeros((W,H,C))
        for key, value in dsecid_to_ddd17id.items():
            label_ddd17[label == key] = value
        img_name = img_path.split('/')[-1]
        img_name = img_name.replace('labelTrainIds','labelTrainIds_ddd17')
        cv2.imwrite(os.path.join(path,img_name),label_ddd17)



