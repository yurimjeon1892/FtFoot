import os
import natsort
import numpy as np

from .utils_loader import pose_read, get_cam_mtx, get_lidar2cam_mtx

def read_raw_sample_list(args):  
    
    seq = args.seq

    calib_dict = {}  
    Tr_fn = os.path.join(args.data_root, "Rellis_3D", str(seq).zfill(5), "transforms.yaml")
    RT = get_lidar2cam_mtx(Tr_fn)
    calib_dict["Tr"] = RT
    calib_dict["Tr_inv"] = np.linalg.inv(RT)

    P_fn = os.path.join(args.data_root, "Rellis-3D", str(seq).zfill(5), "camera_info.txt")
    P = get_cam_mtx(P_fn)
    P_eye = np.eye(4)
    P_eye[:3,:3] = P
    P_eye = P_eye
    calib_dict["P"] = P_eye
    calib_dict["P_inv"] = np.linalg.inv(P_eye)  

    cam_name_dict_seq = {}
    file_list = os.listdir(os.path.join(args.data_root, "Rellis-3D", str(seq).zfill(5), "pylon_camera_node"))
    for file_one in file_list:
        fn = file_one.split("/")[-1]
        cam_name_dict_seq[fn[5:11]] = fn[:-4]

    sample_list = []       

    seq_str = str(seq).zfill(5)   
    f = open(os.path.join(args.data_root, "Rellis-3D", seq_str, "poses.txt"), 'r')
    poses_all = f.readlines()
    f.close()

    file_list = os.listdir(os.path.join(args.data_root, "Rellis-3D", seq_str, "os1_cloud_node_kitti_bin"))
    file_list = natsort.natsorted(file_list)[args.start_num:args.end_num]
    for fn in file_list:
        int_frame = int(fn[:-4])
        
        # pose of i
        P_oi = pose_read(poses_all[int_frame])
        poses_tmp = []
        for fr_n in range(int_frame, min(int_frame + 150, len(poses_all))):
            P_oj = pose_read(poses_all[fr_n])
            P_ij = np.linalg.inv(P_oi) @ P_oj # 4x4
            poses_tmp.append(P_ij)

        str_frame = str(int_frame).zfill(6)
        indiv_sample = {"image": os.path.join(args.data_root, "Rellis-3D", seq_str, "pylon_camera_node", cam_name_dict_seq[str_frame] + ".jpg"),
                        "point": os.path.join(args.data_root, "Rellis-3D", seq_str, "os1_cloud_node_kitti_bin", str_frame + '.bin'),
                        "label": os.path.join(args.data_root, "Rellis-3D", seq_str, "os1_cloud_node_semantickitti_label_id", str_frame + '.label'),       
                        "P_oi": P_oi,
                        "fnum": int_frame}
        sample_list.append(indiv_sample)
        
    return sample_list, calib_dict

semkitti_label_to_color = {
    0: [0, 0, 0], # void
    1: [108, 64, 20], # dirt
    3: [0, 102, 0], # grass
    4: [0, 255, 0], # tree
    5: [0, 153, 153], # pole
    6: [0, 128, 255], # water
    7: [0, 0, 255], # sky
    8: [255, 255, 0], # vehicle
    9: [255, 0, 127], # objecbt
    10: [64, 64, 64], # aphalt
    12: [255, 0, 0], # building
    15: [102, 0, 0], # log
    17: [204, 153, 255], # person
    18: [102, 0, 204], # fence
    19: [255, 153, 204], # bush
    23: [170, 170, 170], # concrete
    27: [41, 121, 255], # barrier
    29: [101, 31, 255], # uphill
    30: [137, 149, 9], # downhill
    31: [134, 255, 239], # puddle
    33: [99, 66, 34], # mud
    34: [110, 22, 138], # rubble
}

# 0 -- Background: void, sky, # brown
# 1 -- Level1 - Navigable: concrete, asphalt @ beige
# 2 -- Level2 - Navigable: dirt, grass, @ green
# 3 -- Level3 - Navigable: mud, rubble @ green
# 4 -- Non-Navigable: water, bush, puddle, #
# 5 -- Obstacle: tree, pole, vehicle, object, building, log, person, fence, barrier

semkitti_color_to_label = {
    "108/64/20": 0,       
    "255/229/204": 1,
    "0/102/0": 2,
    "0/255/0": 3,
    "0/153/153": 4,
    "0/128/255": 5,
}
