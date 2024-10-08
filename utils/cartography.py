from demo.validation import Evaluator
import cv2
import numpy as np 
import os
import glob
from tqdm import tqdm

palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [255, 0, 0],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
            [0, 80, 100], [0, 0, 230], [119, 11, 32]]

def visual_label(label_gt: np.ndarray):
    W, H, C = label_gt.shape
    for w in range(W):
        for h in range(H):
            if label_gt[w,h,0] == 255:
                label_gt[w,h,:] = [255,255,255]
            else:
                label_gt[w,h,:] = palette[label_gt[w,h,0]]
    return cv2.cvtColor(label_gt, cv2.COLOR_BGR2RGB)

def visual_event(event):  
    event = event.sum(axis=0)
    event_r = np.zeros(event.shape)
    event_g = np.zeros(event.shape)
    event_b = np.zeros(event.shape)
    event_r[event > 0] = 255
    event_g[event < 0] = 255
    return np.stack((event_b, event_g, event_r),axis=2).astype(np.uint8)


def test_score():
    evaluator = Evaluator(11)
    score_e2vid = list()
    score_ess = list()
    score_ours = list()

    # miou select
    print("Test E2vid")
    for i in tqdm(range(4021)):
        predict_label = cv2.imread(E2vidLabel_path+'/{:04d}.png'.format(i))[:,:,0]
        predict_label = cv2.resize(predict_label,(640,440))
        gt_label = cv2.imread(GT_path+'/{:04d}.png'.format(i))[:,:,0]
        evaluator.add_batch(gt_label, predict_label)
        score_e2vid.append(evaluator.Mean_Intersection_over_Union())
        evaluator.reset()
    
    print("Test ess")
    for i in tqdm(range(4021)):
        predict_label = cv2.imread(ess_path+'/{:04d}.png'.format(i))[:,:,0]
        gt_label = cv2.imread(GT_path+'/{:04d}.png'.format(i))[:,:,0]
        evaluator.add_batch(gt_label, predict_label)
        score_ess.append(evaluator.Mean_Intersection_over_Union())
        evaluator.reset()

    print("Test ours")
    for i in tqdm(range(4021)):
        predict_label = cv2.imread(ours_path+'/{:04d}.png'.format(i))[:,:,0]
        gt_label = cv2.imread(GT_path+'/{:04d}.png'.format(i))[:,:,0]
        evaluator.add_batch(gt_label, predict_label)
        score_ours.append(evaluator.Mean_Intersection_over_Union())
        evaluator.reset()

if __name__ == "__main__":
    Event_path = '/mnt/workspace/dingyiming/Codes/semi-supervised-uda/data/DSEC_Semantic_e2vid_offline/event/train'
    E2vid_path = '/mnt/workspace/dingyiming/Codes/semi-supervised-uda/data/DSEC_Semantic_e2vid_online/gt_fine/train'
    E2vidLabel_path = '/mnt/workspace/dingyiming/DataBase/Label_visualize/E2vid_label'
    ess_path = '/mnt/workspace/dingyiming/DataBase/Label_visualize/ess_fake'
    ours_path = '/mnt/workspace/dingyiming/DataBase/Label_visualize/ours'
    GT_path = '/mnt/workspace/dingyiming/DataBase/DSEC_raw/gt_fine/train'

    score_e2vid = np.load('./score_e2vid.npy')
    score_ess = np.load('./score_ess.npy')
    score_ours = np.load('./score_ours.npy')
    selected_name = np.where((score_ours - score_ess) > 0.25)[0]
    name_list = np.random.choice(selected_name,4,replace=True).tolist()

    for idx, item in enumerate(name_list):
        name_list[idx] = '{:04d}'.format(item) + '.png'
    # name_list = ['00068.png', '02276.png', '02297.png', '02807.png', '03663.png']


    for name in name_list:
        cv2.imwrite('./event_'+name,visual_event(np.load(os.path.join(Event_path, name.split('.')[0]+'.npy'))[0:9,:,:]))
        cv2.imwrite('./E2vid_'+name,cv2.imread(os.path.join(E2vid_path, name)))
        cv2.imwrite('./E2vidLabel_'+name,visual_label(cv2.imread(os.path.join(E2vidLabel_path, name))))
        cv2.imwrite('./ess_'+name,visual_label(cv2.imread(os.path.join(ess_path, name))))
        cv2.imwrite('./ours_'+name,visual_label(cv2.imread(os.path.join(ours_path, name))))
        cv2.imwrite('./GT_'+name,visual_label(cv2.imread(os.path.join(GT_path, name))))



