import os
import random
import numpy as np
import math


# root_sample = './data/DSEC_Semantic_e2vid_online/event/train'
# root_mask = './data/DSEC_Semantic_e2vid_online/gt_fine/train'

root_sample = '/mnt/petrelfs/jinglinglin/Event_segmentation3/semi-supervised-uda-v2/data/DSEC_Semantic_e2vid_online/event/train'
root_mask = '/mnt/petrelfs/jinglinglin/Event_segmentation3/semi-supervised-uda-v2/data/DSEC_Semantic_e2vid_online/gt_fine/train'
file_names = os.listdir(root_sample)

a = int(len(file_names) / 1)

c = file_names[0]
sample = list(np.random.choice(file_names , a+1))

file_names *= math.ceil(30000 / len(file_names))
file_names = file_names[0:30000]


# for i in range (len(file_names)):
#     if file_names[i] not in sample:
#         f = open('/mnt/petrelfs/jinglinglin/Event_segmentation3/semi-supervised-uda-v2/data/DSEC_Semantic_e2vid_online/desc_unlabel_all.txt', 'a')
#         # path = os.path.join(root_sample, file_names[i])
#         # path2 = os.path.join(root_mask, file_names[i])
#         name_idx = file_names[i].split(".")[0]
#         f.write(name_idx + '\n')
#         # f.write(path2 + '\n')
#         f.close()

sample *= math.ceil(25000 / len(sample))
sample = sample[0:25000]


for j in range(len(sample)):
    f = open('/mnt/petrelfs/jinglinglin/Event_segmentation3/semi-supervised-uda-v2/data/DSEC_Semantic_e2vid_online/desc_label_all.txt', 'a')
    # path = os.path.join(root_sample, file_names[j])
    # path2 = os.path.join(root_mask, file_names[j])
    name_idx = sample[j].split(".")[0]
    f.write(name_idx + '\n')
    # f.write(path2 + '\n')
    f.close()
