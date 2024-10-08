import cv2 
import numpy as np
import os
import torch

def trans_event(event):  
                    event = event.sum(dim=0)
                    event_r = torch.zeros(event.shape)
                    event_g = torch.zeros(event.shape)
                    event_b = torch.zeros(event.shape)
                    event_r[event > 0] = 255
                    event_g[event < 0] = 255
                    return torch.stack((event_b, event_g, event_r)).type(torch.uint8)

if __name__ == "__main__":
    palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                [107, 142, 35], [152, 251, 152], [70, 130, 180], [255, 0, 0],
                [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    # event_root = "/mnt/workspace/dingyiming/Codes/semi-supervised-uda/data/DSEC_Semantic/event/train"
    gray_root = "/mnt/workspace/dingyiming/Codes/semi-supervised-uda/data/DSEC_Semantic_e2vid_online/gt_fine/train_pro"
    # label_root = "/mnt/workspace/dingyiming/Codes/DDD17_frame/abandoned/ddd17_raw/gt_fine/train"
    label_gt_root = "/mnt/workspace/dingyiming/Codes/semi-supervised-uda/output"
    label_list = ["0001.png","0005.png","0010.png"]

    for label_name in label_list:
        # label = cv2.imread(os.path.join(label_root, label_name))
        # W, H, C = label.shape
        # for w in range(W):
        #     for h in range(H):
        #         if label[w,h,0] == 255:
        #             label[w,h,:] = [255,255,255]
        #         else:
        #             label[w,h,:] = palette[label[w,h,0]]
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        label_gt = cv2.imread(os.path.join(label_gt_root, label_name))
        W, H, C = label_gt.shape
        for w in range(W):
            for h in range(H):
                if label_gt[w,h,0] == 255:
                    label_gt[w,h,:] = [255,255,255]
                else:
                    label_gt[w,h,:] = palette[label_gt[w,h,0]]
        label_gt = cv2.cvtColor(label_gt, cv2.COLOR_BGR2RGB)

        # event = np.load(os.path.join(event_root,label_name.replace('.png','.npy')))
        # event = trans_event(torch.tensor(event[0:9,:,:])).permute(1,2,0).numpy()

        gray = cv2.imread(os.path.join(gray_root, label_name))

        # cv2.imwrite("./event_"+ label_name, event)
        cv2.imwrite("./gray_"+ label_name, gray)
        # cv2.imwrite("./visual_"+ label_name, label)
        cv2.imwrite("./visual_gt_"+ label_name, label_gt)
