import os
import numpy as np
import math

# Setting
root_sample = '/mnt/petrelfs/jinglinglin/Event_segmentation3/semi-supervised-uda-v2/data/DSEC_Semantic_e2vid_online/event/train'
file_names = os.listdir(root_sample)
percentage = 6

# Sample
data_size = len(file_names)
select_data_size = int(data_size / 100 * percentage)
select_file_names = list(np.random.choice(file_names, select_data_size, replace=True))
assert len(select_file_names) == select_data_size
print("Dataset total number: {:d}".format(data_size))
print("Selected number: {:d}".format(select_data_size))

# Align to GTA
gta_size = 30000
counter = 0

# Create unlabeled txt file
with open('/mnt/petrelfs/jinglinglin/Event_segmentation3/semi-supervised-uda-v2/data/DSEC_Semantic_e2vid_online/label6.txt', 'w') as f:
    while 1:
        for file_name in file_names:
            if file_name not in select_file_names:
                name_idx = file_name.split(".")[0]
                f.write(name_idx + '\n')
                counter += 1
                if counter >= gta_size:
                    break
        if counter >= gta_size:
            break
    counter = 0

# Create labeled txt file
with open('/mnt/petrelfs/jinglinglin/Event_segmentation3/semi-supervised-uda-v2/data/DSEC_Semantic_e2vid_online/unlabel6.txt', 'w') as f:
    while 1:
        for file_name in select_file_names:
            name_idx = file_name.split(".")[0]
            f.write(name_idx + '\n')
            counter += 1
            if counter >= gta_size:
                break
        if counter >= gta_size:
            break
    counter = 0



