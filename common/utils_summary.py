# helper functions for training
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def summary_image_draw(rgbd, gts, preds, cmap=plt.cm.plasma):

    rgb_np = rgbd.cpu().detach().numpy()[0][:3, :, :]
    rgb_np = np.transpose(rgb_np, (1, 2, 0)) * 255
    rgb_np = rgb_np.astype('uint8') 

    d_np = rgbd.cpu().detach().numpy()[0][3, :, :]
    d_np = minmax_color_img_from_img_numpy(d_np, cmap)
    
    input_img = np.concatenate([rgb_np, d_np], 1)

    sn_np = gts["sn"].cpu().detach().numpy()[0][:3, :, :]
    sn_np = (np.transpose(sn_np, (1, 2, 0)) + 1) * 127
    sn_np = sn_np.astype('uint8') 

    sp_np = gts["sp"].cpu().detach().numpy()[0][0, :, :]
    sp_np = minmax_color_img_from_img_numpy(sp_np, cmap)

    trav_np = gts["fp"].cpu().detach().numpy()[0][0, :, :]
    trav_np = minmax_color_img_from_img_numpy(trav_np, cmap)
    
    gt_img = np.concatenate([sn_np, sp_np, trav_np], 1)        

    pred_sn_np = preds["sn"].cpu().detach().numpy()[0][:3, :, :]
    pred_sn_np = (np.transpose(pred_sn_np, (1, 2, 0)) + 1) * 127
    pred_sn_np = pred_sn_np.astype('uint8')     

    pred_trav_np = preds["trav"].softmax(dim=1)
    pred_trav_np = pred_trav_np.cpu().detach().numpy()[0][-1, :, :]    
    # pred_trav_np = np.argmax(pred_trav_np, axis=1)[0]
    pred_trav_np = minmax_color_img_from_img_numpy(pred_trav_np, cmap)
    pred_trav_np = np.array(Image.fromarray(pred_trav_np).resize((pred_sn_np.shape[1], pred_sn_np.shape[0])))   
    
    pred_img = np.concatenate([pred_sn_np, pred_trav_np], 1)

    summary_img = {}
    summary_img["0_input"] = input_img   
    summary_img["1_gt"] = gt_img
    summary_img["2_pred"] = pred_img
    
    if "gt" in gts.keys():
        gt_np = gts["gt"].cpu().detach().numpy()[0][0, :, :]
        gt_np = minmax_color_img_from_img_numpy(gt_np, cmap)
        summary_img["4_orfd"] = gt_np
    
    return summary_img

def summary_image_draw_self_supervise(rgbd, preds, cmap=plt.cm.plasma):

    rgb_np = rgbd.cpu().detach().numpy()[0][:3, :, :]
    rgb_np = np.transpose(rgb_np, (1, 2, 0)) * 255
    rgb_np = rgb_np.astype('uint8') 

    d_np = rgbd.cpu().detach().numpy()[0][3, :, :]
    d_np = minmax_color_img_from_img_numpy(d_np, cmap)
    
    input_img = np.concatenate([rgb_np, d_np], 1)

    pred_sn_np = preds["sn"].cpu().detach().numpy()[0][:3, :, :]
    pred_sn_np = (np.transpose(pred_sn_np, (1, 2, 0)) + 1) * 127
    pred_sn_np = pred_sn_np.astype('uint8') 

    pred_trav_np = preds["trav"].softmax(dim=1)
    pred_trav_np = pred_trav_np.cpu().detach().numpy()[0][-1, :, :]  
    # pred_trav_np = np.argmax(pred_trav_np, axis=1)[0]
    pred_trav_np = minmax_color_img_from_img_numpy(pred_trav_np, cmap)
    pred_trav_np = np.array(Image.fromarray(pred_trav_np).resize((pred_sn_np.shape[1], pred_sn_np.shape[0])))   

    pred_img = np.concatenate([pred_sn_np, pred_trav_np], 1)
    
    summary_img = {}
    summary_img["3_input"] = input_img   
    summary_img["3_pred"] = pred_img
    
    # for k in summary_img.keys():
    #     print(k, summary_img[k].shape)

    return summary_img
    
def update_summary(summary, rgbd, gts, pred, lss_dict, rgbd_s, pred_s, epoch, mode):

    for k in lss_dict.keys():
        metric_k = np.clip(lss_dict[k], 0.0, 100.0)
        summary.add_scalar(mode + "_metric/" + k, metric_k, epoch) 
    
    summary_img = summary_image_draw(rgbd, gts, pred) 
    for k in list(summary_img.keys()):
        summary_img_k = summary_img[k]
        if summary_img_k.shape[2] == 3:
            summary_img_k = np.transpose(summary_img[k], (2, 0, 1))
        summary.add_image(mode + '_image/' + k, summary_img_k, epoch)    

    summary_img = summary_image_draw_self_supervise(rgbd_s, pred_s)
    for k in list(summary_img.keys()):
        summary_img_k = summary_img[k]
        if summary_img_k.shape[2] == 3:
            summary_img_k = np.transpose(summary_img[k], (2, 0, 1))
        summary.add_image(mode + '_image/' + k, summary_img_k, epoch)   

    return

def minmax_color_img_from_img_numpy(img, cmap, px=1, valid_mask=False):
    """
    :param img: Input image (numpy array, H x W)
    :param cmap: plt color map
    :param px: pixel size (int)
    :param valid_mask: return valida mask? (bool)
    :return img: minmax colored image (numpy array, H x W x 3)
    """
    if np.max(img) > np.min(img):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  
    else: img = np.zeros_like(img)
    height, width = img.shape[0], img.shape[1]
    minmax_img = np.zeros(shape=(height, width))
    for y in range(height):
        for x in range(width):
            if img[y, x] > 0:
                y_min, y_max = np.maximum(0, y - px), np.minimum(height - 1, y + px + 1)
                x_min, x_max = np.maximum(0, x - px), np.minimum(width - 1, x + px + 1)
                max_depth = np.max(minmax_img[y_min:y_max, x_min:x_max])
                if max_depth < img[y, x]:
                    minmax_img[y_min:y_max, x_min:x_max] = img[y, x]
    v_mask = (minmax_img > 0).reshape(minmax_img.shape)
    minmax_img = 255 * cmap(minmax_img)[:, :, :3]
    minmax_img = minmax_img.astype('uint8')
    if valid_mask: return minmax_img, v_mask
    else: return minmax_img

