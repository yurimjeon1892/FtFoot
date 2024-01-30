import os
import argparse
import natsort
import numpy as np
from tqdm import tqdm
from common.utils_loader import bin_img_read
from common.utils import resize_rgb_image

def confusion_matrix_orfd(x, y, n, ignore_label=None, mask=None):
    if mask is None:
        mask = np.ones_like(x) == 1
    k = (x >= 0) & (y < n) & (x != ignore_label) & (mask.astype(np.bool_))
    return np.bincount(n * x[k].astype(int) + y[k], minlength=n**2).reshape(n, n)

def getScores(conf_matrix):
    if conf_matrix.sum() == 0:
        return 0, 0, 0, 0, 0
    with np.errstate(divide='ignore',invalid='ignore'):
        globalacc = np.diag(conf_matrix).sum() / conf_matrix.sum()
        classpre = np.diag(conf_matrix) / conf_matrix.sum(0)
        classrecall = np.diag(conf_matrix) / conf_matrix.sum(1)
        IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix))
        pre = classpre[1]
        recall = classrecall[1]
        iou = IU[1]
        F_score = 2*(recall*pre)/(recall+pre)
    return globalacc, pre, recall, F_score, iou

if __name__ == '__main__':
    
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Evaluate the predictions of orfd")
    parser.add_argument("--data_root", type=str, help="Data root", required=True)
    parser.add_argument("--pred_path", type=str, help="Prediction path", required=True)

    # Parse the command line arguments to an object
    args = parser.parse_args()

    print("*********************************************")
    print("[i] start evaluation")
    print("[i] data root: ", args.data_root)
    print("[i] pred path: ", args.pred_path)
    
    num_labels = 2
    
    file_list = natsort.natsorted(os.listdir(args.pred_path))
    conf_mat = np.zeros((num_labels, num_labels), dtype=np.float16)
    for i, fname in enumerate(tqdm(file_list)):
        fn = fname[:-4]
        # size: 1, 720, 1280, min 0 max 1
        gt_ = bin_img_read(os.path.join(args.data_root, "Final_Dataset", "testing", "gt_image", fn + "_fillcolor.png"))
        pred_ = bin_img_read(os.path.join(args.pred_path, fname))[:, :, 0]
        pred_ = resize_rgb_image(pred_, (720, 1280))
        
        gt_ = np.expand_dims(gt_, 0)
        pred_ = np.expand_dims(pred_, 0)
        
        print(gt_.shape, np.max(gt_), np.min(gt_), pred_.shape, np.max(pred_), np.min(pred_))
            
        conf_mat_ = confusion_matrix_orfd(gt_, pred_, num_labels)   
        conf_mat = conf_mat + conf_mat_
        
    globalacc, pre, recall, F_score, iou = getScores(conf_mat)
    
    print ('glob acc : {0:.3f}, pre : {1:.3f}, recall : {2:.3f}, F_score : {3:.3f}, IoU : {4:.3f}'.format(globalacc, pre, recall, F_score, iou)) 
    d = '{0:.3f}, {1:.3f}, {2:.3f}, {3:.3f}, {4:.3f}\n'.format(globalacc, pre, recall, F_score, iou)  
    
    f = open("eval_orfd_score.txt", "a")
    f.write("method, glob acc, pre, recall, F_score, IoU\n")
    f.write(args.pred_path.split("/")[-1] + ", " + d)
    f.close()
