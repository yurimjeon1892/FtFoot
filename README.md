# FtFoot : Follow the Footprints :footprints:
This repository contains the code (in PyTorch) for "Follow the Footprints: Self-supervised Traversability Estimation for Off-road Vehicle Navigation based on Geometric and Visual Cues" paper (ICRA 2024).

## Environment

### Step 1: Requirements
* CUDA 11.3
* cuDNN 8
* Ubuntu 20.04

### Step 2: Create conda environment
```
conda create -n ftfoot python=3.8
conda activate ftfoot
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Step 3: Modify torch-encoding script
Modify the torch-encoding script by referring to [this](https://github.com/zhanghang1989/PyTorch-Encoding/issues/328#issuecomment-749549857):

> `cd anaconda3/envs/ftfoot/lib/python3.8` # this path can be different, depends on your environment
> 
> 1.  fix the code in `site-packages/encoding/nn/syncbn.py` at about line 200   
> from  
> `return syncbatchnorm(........).view(input_shape)`  
> to  
> `x, _, _=syncbatchnorm(........)`  
> `x=x.view(input_shape)`  
> `return x`  
> 
> 2.  fix the code `site-packages/encoding/function/syncbn.py` at about line 102  
> from  
> `ctx.save_for_backward(x,_ex,_exs,gamma,beta)`    
> `return y`    
> to  
> `ctx.save_for_backward(x,_ex,_exs,gamma,beta)`   
> `ctx.mark_non_differentiable(running_mean,running_var)`  
> `return y,running_mean,running_var`  
> 
> 3.  fix the code `site-packages/encoding/function/syncbn.py` at about line 109  
> from  
> `def backward(ctx,dz):`  
> to  
> `def backward(ctx,dz,_druning_mean,_druning_var):`  

### Step 4: Install GFL
```
cd exts
python setup.py install
```

## Data

### RELLIS-3D

1. Download [RELLIS-3D dataset](https://unmannedlab.github.io/research/RELLIS-3D). The folder structure is as follows. 
```
RELLIS-3D
├── Rellis-3D
|   ├── 00000
|   |   ├── os1_cloud_node_kitti_bin
|   |   ├── pylon_camera_node
|   |   ├── calib.txt
|   |   ├── camera_info.txt
|   |   └── poses.txt    
|   ├── 00001
|   └── ..
└── Rellis_3D
    ├── 00000
    |   └── transforms.yaml
    ├── 00001
    └── ..
```

2. Prepare the data for training. Run:
```
sh ./data_prep/rellis_preproc.sh
```

3. The final folder structure is as follows. 
```
RELLIS-3D
├── Rellis-3D
├── Rellis_3D
└── Rellis-3D-custom
    ├── 00000
    |   ├── foot_print
    |   ├── super_pixel # This is optional, but recommended for clear output!
    |   └── surface_normal
    ├── 00001
    └── ..
```

### ORFD

1. Download [ORFD dataset](https://github.com/chaytonmin/Off-Road-Freespace-Detection). The folder structure is as follows. 
```
ORFD
└── Final_Dataset
    ├── training
    |   ├── calib
    |   ├── dense_depth
    |   ├── gt_image
    |   ├── image_data
    |   ├── lidar_data
    |   └── sparse_depth    
    ├── validation
    └── testing
```

2. This dataset has no pose data. Therefore, we need to estimate the pose data from the point cloud. We used [PyICP-SLAM](https://github.com/gisbi-kim/PyICP-SLAM). Place the pose data under the directory.
```
ORFD
├── Final_Dataset
└── ORFD-custom
    ├── training
    |   └── pose
    |       └── pose_16197787.csv
    ├── validation
    └── testing
```

3. Prepare the data for training. Run:
```
sh ./data_prep/orfd_preproc.sh
```

4. The final folder structure is as follows. 
```
ORFD
├── Final_Dataset
└── ORFD-custom
    ├── training
    |   ├── foot_print
    |   ├── pose
    |   ├── super_pixel
    |   └── surface_normal
    ├── validation
    └── testing
```

## Traversability Estimation

### Train 
Set `data_config/data_root` in the `train.yaml` file and run:
```
python train.py configs/train.yaml
```

### Test
Set `data_config/data_root` and `resume_path` in the `test.yaml` file and run:
```
python test.py configs/test.yaml
```

## Path Plan

### Plot global cost map
```
python ./plot_map/plot_map_rellis.py \
        --start_num 400 --end_num 900 \
        --save_rgb_img --save_valid_map \
        --cost_path ../outputs/prediction/your-ckpt-name
```

### Generate path

```
python ./path_plan/path_plan.py \
        --start_num 400 --end_num 900 \
        --local_planner_type TRRTSTAR \
        --max_path_iter 1000 --max_extend_length 10 --bias_sampling \
        --cost_map_name /path/to/your-ckpt-name.png

```

## Acknowledgements
* Our implementation of GFL is based on https://github.com/kakaxi314/GuideNet
* Our implementation of FSM is based on https://github.com/panzhiyi/URSS

If you have any questions, please contact Yurim Jeon at yurimjeon1892@gmail.com
