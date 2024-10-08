import glob
import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

class val_loader(Dataset):
    def __init__(self, selected_list=None):
        self.label_predict_path = "./output"
        self.label_gt_path = "/mnt/workspace/dingyiming/DataBase/DSEC_raw/gt_fine/train"

        if selected_list == None:
            self.label_predict_list = sorted(glob.glob(self.label_predict_path + '/*.*'))
            self.label_gt_list = sorted(glob.glob(self.label_gt_path + '/*.*'))
        else:
            assert isinstance(selected_list, str)
            self.label_predict_list = list()
            self.label_gt_list = list()

            with open(selected_list,'r') as f:
                for file_name in f.readlines():
                    file_name = file_name.replace('\n','.png')
                    self.label_predict_list.append(os.path.join(self.label_predict_path,file_name))
                    self.label_gt_list.append(os.path.join(self.label_gt_path,file_name))

        self.length = self.label_predict_list.__len__()

    def __getitem__(self, index):
        item_predict = cv2.imread(self.label_predict_list[index % self.length])[:,:,0]
        item_label   = cv2.imread(self.label_gt_list[index % self.length])[:,:,0]

        return item_predict, item_label

    def __len__(self):
        return self.length

def draw_cm():
    conf_matrix = np.load('./confusion_matrix.npy')
    # 绘制混淆矩阵
    labels =   ['Sky',
                'Bui.',
                'Fence',
                'Person',
                'Pole',
                'Road',
                'Sidewalk', 
                'Veg.',
                'Trans.',
                'Wall',
                'Traffic']

    # 显示数据
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    # 在图中标注数量/概率信息
    thresh = conf_matrix.max() / 2	#数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(classes):
        for y in range(classes):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = round(conf_matrix[y, x],2)
            plt.text(x, y, info,
                    verticalalignment='center',
                    horizontalalignment='center',
                    color="white" if info > thresh else "black")
                    
    plt.tight_layout()
    plt.yticks(range(classes), labels)
    plt.xticks(range(classes), labels,rotation=45)#X轴字体倾斜45°
    plt.show()
    plt.savefig('./mc.png', dpi=300)
    plt.close()


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2) #21*21的矩阵,行代表ground truth类别,列代表preds的类别,值代表
 
    '''
    正确的像素占总像素的比例
    '''
    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc
 
    '''
    分别计算每个类分类正确的概率
    '''
    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU) #跳过0值求mean,shape:[21]
        return MIoU
 
    def Class_IOU(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return MIoU
 
    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
 
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
 
 
    '''
    参数的传入:
        evaluator = Evaluate(4)           #只需传入类别数4
        evaluator.add_batch(target, preb) #target:[batch_size, 512, 512]    ,    preb:[batch_size, 512, 512]
        在add_batch中统计这个epoch中所有图片的预测结果和ground truth的对应情况, 累计成confusion矩阵(便于之后求mean)
    
    
    参数列表对应:
        gt_image: target  图片的真实标签            [batch_size, 512, 512]
        pre_image: pred   网络生成的图片的预测标签   [batch_size, 512, 512]
    '''
    # 计算混淆矩阵
    def _generate_matrix(self, gt_image, pre_image):
        assert pre_image.max() < self.num_class
        mask = (gt_image >= 0) & (gt_image < self.num_class)       #ground truth中所有正确(值在[0, classe_num])的像素label的mask
        
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask] 
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix
 
    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        tmp = self._generate_matrix(gt_image, pre_image)
        #矩阵相加是各个元素对应相加,即21*21的矩阵进行pixel-wise加和
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
 
 
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def report(self):
        TP = self.confusion_matrix.diagonal()
        FP = self.confusion_matrix.sum(axis=0) - TP
        FN = self.confusion_matrix.sum(axis=1) - TP
        TN = self.confusion_matrix.sum() - \
                (self.confusion_matrix.sum(axis=0) +\
                self.confusion_matrix.sum(axis=1) - TP)
        # 查准率(precision)
        P = TP / (TP + FP)
        # 查全率(recall)
        R = TP / (TP + FN)
        print("Class IOU")
        print((self.Class_IOU()*100).tolist())
        print("Mean_Intersection_over_Union")
        print(self.Mean_Intersection_over_Union()*100)
        print("查准率(precision)")
        print(P)
        print("查全率(recall)")
        print(R)

if __name__ == '__main__':
    test_loader = DataLoader(val_loader(selected_list=None), batch_size=1, shuffle=False, num_workers=15)
    # 定义一个 分类数*分类数 的空混淆矩阵
    trick = False
    evaluator = Evaluator(11)
    score = list()
    score_threshold = 0

    class_fence = list()
    class_Pole = list()
    class_Wall = list()

    print("Testing........")
    with torch.no_grad():
        for predicts, targets in tqdm(test_loader):
            # 更新混淆矩阵
            targets = targets.numpy()
            predicts = predicts.numpy()
            evaluator.add_batch(targets, predicts)

            if trick:
                score.append(evaluator.Mean_Intersection_over_Union())
                evaluator.reset()

        if trick:
            score = np.array(score)

            score_indice = np.where(score > score_threshold)[0]
            with open('./desc_label_'+str(int(score_threshold*100))+'.txt', 'w') as f:
                for index in score_indice:
                    f.write('{:05d}'.format(index)+'\n')

            score_indice = np.where(score <= score_threshold)[0]
            with open('./desc_unlabel_'+str(int(score_threshold*100))+'.txt', 'w') as f:
                for index in score_indice:
                    f.write('{:05d}'.format(index)+'\n')

        if not trick:
            TP = evaluator.confusion_matrix.diagonal()
            FP = evaluator.confusion_matrix.sum(axis=0) - TP
            FN = evaluator.confusion_matrix.sum(axis=1) - TP
            TN = evaluator.confusion_matrix.sum() - \
                 (evaluator.confusion_matrix.sum(axis=0) +\
                  evaluator.confusion_matrix.sum(axis=1) - TP)
            # 查准率(precision)
            P = TP / (TP + FP)
            # 查全率(recall)
            R = TP / (TP + FN)
            print("Class IOU")
            print((evaluator.Class_IOU()*100).tolist())
            print("Mean_Intersection_over_Union")
            print(evaluator.Mean_Intersection_over_Union()*100)
            print("查准率(precision)")
            print(P)
            print("查全率(recall)")
            print(R)




