**News**
* `25/01/2022` To make the comparison with our Neural Human Performer easier, we provide the evaluation results (images, summary) at [here](https://drive.google.com/file/d/1ZV300Aukl4LuTy-65qr9Oo3d2TytrcNP/view?usp=sharing). 
* `25/01/2022` The code and the pretrained model for the Neural Human Performer are now released!
# Neural Human Performer: Learning Generalizable Radiance Fields for Human Performance Rendering
### [Project Page](https://youngjoongunc.github.io/nhp/) | [Video](https://www.youtube.com/watch?v=4b5SPwPOKVo) | [Paper](https://arxiv.org/pdf/2109.07448.pdf)


<img src="https://github.com/YoungJoongUNC/Neural_Human_Performer/blob/main/image/teaser.gif?raw=true" width="70%" height="70%" />

> [Neural Human Performer: Learning Generalizable Radiance Fields for Human Performance Rendering](https://arxiv.org/pdf/2012.15838.pdf)  
> Youngjoong Kwon, Dahun Kim, Duygu Ceylan, Henry Fuchs  
> NeurIPS 2021 (Spotlight)



## Installation

Please see [INSTALL.md](INSTALL.md).
We provide the pretrained models at [here](https://drive.google.com/file/d/1EyqrzHZws_A9ixY3RAxN82WBL1FgypFj/view?usp=sharing)

## Run the code on ZJU-MoCap

Please see [INSTALL.md](INSTALL.md) to download the dataset.

The provided commands are for the evaluation and visualization on the unseen subjects (subject 387, 393, 394)
with three input cameras (Camera B1, B8, B16). To make the comparison with our Neural Human Performer easier, 
we provide the evaluation results (images, summary) at [here](https://drive.google.com/file/d/1ZV300Aukl4LuTy-65qr9Oo3d2TytrcNP/view?usp=sharing). 
* To train / test on the different subjects, please modify the `lib/datasets/get_human_info.py` file. 
* To use different input views, please modify the `test_input_view` command argument.
* To test on the different setting (e.g., test on unseen poses of seen subjects), 
please modify the `test_mode` command argument accordingly. There are four different settings: 
`model_o_motion_o` `model_o_motion_x` `model_x_motion_o` `model_x_motion_x`


### Evaluation on ZJU-MoCap
1. Download the pretrained model and put it to `$ROOT/data/trained_model/if_nerf/demo/latest.pth`.
2. Quantitative evaluation:
    ```
    CUDA_VISIBLE_DEVICES=0 python run.py --type evaluate --cfg_file configs/train_or_eval.yaml virt_data_root data/zju_mocap rasterize_root data/zju_rasterization ratio 0.5 H 1024 W 1024 test_input_view "0,7,15" run_mode test test_mode model_x_motion_x exp_name demo resume True test_sample_cam True test.epoch -1 exp_folder_name debug gpus "0,"
    ```
3. The results (images, summary) will be saved at `$ROOT/data/result/if_nerf/{$exp_name}/epoch_{$test.epoch}/{$exp_folder_name}`

### Visualization on ZJU-MoCap
1. Download the corresponding pretrained model and put it to `$ROOT/data/trained_model/if_nerf/demo/latest.pth`.
2. Visualization:
    * Free-viewpoint rendering
    ```
    # render frames
    # results will be saved at '$ROOT/data/perform/{$exp_name}/epoch_{$test.epoch}/{$exp_folder_name}'
    CUDA_VISIBLE_DEVICES=0 python run.py --type visualize --cfg_file configs/performance.yaml virt_data_root data/zju_mocap rasterize_root data/zju_rasterization ratio 0.5 H 1024 W 1024 test_input_view "0,7,15" test_mode model_x_motion_x exp_name demo resume True test.epoch -1 exp_folder_name debug gpus "0,"
    
    # generate video
    python gen_freeview_video.py 
    ```
    <p float="left">  
      <img src="https://github.com/YoungJoongUNC/Neural_Human_Performer/blob/main/image/freeview_subject_0.gif?raw=true" width="30%" height="30%" />
      <img src="https://github.com/YoungJoongUNC/Neural_Human_Performer/blob/main/image/freeview_subject_1.gif?raw=true" width="30%" height="30%" />
      <img src="https://github.com/YoungJoongUNC/Neural_Human_Performer/blob/main/image/freeview_subject_2.gif?raw=true" width="30%" height="30%" />
    </p>

    * Mesh recosntruction
    ```
    # reconstruct the mesh
    # results will be saved at '$ROOT/data/mesh/{$exp_name}/epoch_{$test.epoch}/{$exp_folder_name}'
    CUDA_VISIBLE_DEVICES=0 python run.py --type visualize --cfg_file configs/reconstruction.yaml virt_data_root data/zju_mocap rasterize_root data/zju_rasterization ratio 0.5 H 1024 W 1024 test_input_view "0, 7, 15" test_mode model_x_motion_x exp_name demo resume True test.epoch -1 exp_folder_name debug gpus "1,"
    
    # render mesh
    # render mesh from 5th frame of 0th human
    export MESA_GL_VERSION_OVERRIDE=3.3
    python tools/render_mesh.py --exp_name demo --epoch -1 --exp_folder_name debug --dataset zju_mocap --human_idx 0 --frame_idx 5
    ```
### Training on ZJU-MoCap


1. Train:
    ```
    # training
    CUDA_VISIBLE_DEVICES=0 python train_net.py --cfg_file configs/train_or_eval.yaml virt_data_root data/zju_mocap rasterize_root data/zju_rasterization ratio 0.5 H 1024 W 1024 run_mode train jitter True exp_name nhp resume True gpus "0,"
    ```

3. Tensorboard:
    ```
    tensorboard --logdir data/record/if_nerf
    ```   

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@article{kwon2021neural,
  title={Neural human performer: Learning generalizable radiance fields for human performance rendering},
  author={Kwon, Youngjoong and Kim, Dahun and Ceylan, Duygu and Fuchs, Henry},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
## Acknowledgments

We thank [Sida Peng](https://pengsida.net/) of Zhejiang University, Hangzhou, China, for 
very many helpful discussions on a variety of implementation details of [Neural Body](https://zju3dv.github.io/neuralbody/).
We thank [Ruilong li](https://www.liruilong.cn/) and [Alex Yu](https://alexyu.net/) of UC Berkeley
for many discussions on the [AIST++](https://google.github.io/aichoreographer/) dataset and [pixelNeRF](https://alexyu.net/pixelnerf/) details.
We thank Prof. [Alex Berg](http://acberg.com/) of UNC for the generous offer of computational resources and [Misha Shvets](https://scholar.google.com/citations?user=CY9PcHEAAAAJ&hl=en) of UNC
for a useful tutorial on it. This work was partially supported by National Science Foundation Award 1840131.
## Contact

For questions, please contact [youngjoong@cs.unc.edu](youngjoong@cs.unc.edu).


## THuman dataset implementation

Naming:
nhp: on-the-fly
thuman_nhp: use full thuman training set and evaluate on testing set
    exp name "perceptual": using perceptual metrics (LPIPS and FID), taking input view [0, 6, 11] and output [3, 8, 13]
thuman_nhp1: use partial thuman training set
thuman_depth: use sapiens depth

train
```
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --cfg_file configs/thuman.yaml \
  virt_data_root data/THuman/train \
  H 1024 W 1024 \
  run_mode train \
  jitter True \
  exp_name thuman_depth \
  resume True \
  gpus "0,"
```


evaluate
```
CUDA_VISIBLE_DEVICES=0 python run.py \
  --type evaluate \
  --cfg_file configs/thuman.yaml \
  virt_data_root data/THuman/val \
  ratio 0.5 \
  H 1024 W 1024 \
  test_input_view "0,6,11" \
  run_mode test \
  test_mode model_x_motion_x \
  exp_name thuman_depth \
  resume True \
  test_sample_cam True \
  test.epoch -1 \
  exp_folder_name debug \
  gpus "0,"
```

Run evaluate for perceptual metrics
```
CUDA_VISIBLE_DEVICES=0 python run.py \
  --type evaluate \
  --cfg_file configs/thuman.yaml \
  virt_data_root data/THuman/val \
  ratio 0.5 \
  H 1024 W 1024 \
  test_input_view "0,6,11" \
  run_mode test \
  test_mode model_x_motion_x \
  exp_name thuman_nhp \
  resume True \
  test_sample_cam True \
  test.epoch -1 \
  exp_folder_name perceptual \
  gpus "0,"
```


* Free-viewpoint rendering
    ```
    # render frames
    # results will be saved at '$ROOT/data/perform/{$exp_name}/epoch_{$test.epoch}/{$exp_folder_name}'
    # change train to val in virt_data_root
    CUDA_VISIBLE_DEVICES=0 python run.py --type visualize --cfg_file configs/thuman_performance.yaml virt_data_root data/THuman/val ratio 0.5 H 1024 W 1024 test_mode model_x_motion_x exp_name thuman_nhp resume True test.epoch -1 exp_folder_name debug gpus "0,"

    
    # generate video
    python gen_freeview_video.py 
    ```


# No SMPL train
# with mask
```
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --cfg_file configs/thuman_no_smpl.yaml \
  virt_data_root data/THuman/train \
  H 1024 W 1024 \
  run_mode train \
  jitter True \
  exp_name thuman_depth_no_smpl \
  resume False \
  gpus "0,"
```

```
CUDA_VISIBLE_DEVICES=0 python run.py \
  --type evaluate \
  --cfg_file configs/thuman_no_smpl.yaml \
  virt_data_root data/THuman/val \
  ratio 0.5 \
  H 1024 W 1024 \
  test_input_view "0,6,11" \
  run_mode test \
  test_mode model_x_motion_x \
  exp_name thuman_depth_no_smpl \
  resume False \
  test_sample_cam True \
  test.epoch 100 \
  exp_folder_name perceptual \
  gpus "0,"
```


## update forward function

```
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --cfg_file configs/thuman_no_smpl.yaml \
  virt_data_root data/THuman/train \
  H 1024 W 1024 \
  run_mode train \
  jitter True \
  exp_name thuman_depth_no_smpl_forward_new \
  resume True \
  gpus "0,"
```

```
CUDA_VISIBLE_DEVICES=0 python run.py \
  --type evaluate \
  --cfg_file configs/thuman_no_smpl.yaml \
  virt_data_root data/THuman/val \
  ratio 0.5 \
  H 1024 W 1024 \
  test_input_view "0,6,11" \
  run_mode test \
  test_mode model_x_motion_x \
  exp_name thuman_depth_no_smpl_forward_new \
  resume True \
  test_sample_cam True \
  test.epoch -1 \
  exp_folder_name perceptual \
  gpus "0,"
```


# 15 query points instead of 5. use smplx for black whole filling
```
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --cfg_file configs/thuman_no_smpl.yaml \
  virt_data_root data/THuman/train \
  H 1024 W 1024 \
  run_mode train \
  jitter True \
  exp_name thuman_depth_no_smpl3 \
  resume True \
  use_smpl_depth True \
  save_depth False \
  gpus "0,"
```

```
CUDA_VISIBLE_DEVICES=0 python run.py \
  --type evaluate \
  --cfg_file configs/thuman_no_smpl.yaml \
  virt_data_root data/THuman/val \
  ratio 0.5 \
  H 1024 W 1024 \
  run_mode test \
  test_mode model_x_motion_x \
  exp_name thuman_depth_no_smpl3 \
  resume True \
  test_sample_cam True \
  test.epoch 30 \
  exp_folder_name perceptual \
  use_smpl_depth True \
  save_depth True \
  gpus "0,"
```


# 4: 31 query points
```
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --cfg_file configs/thuman_no_smpl.yaml \
  virt_data_root data/THuman/train \
  H 1024 W 1024 \
  run_mode train \
  jitter False \
  exp_name thuman_depth_no_smpl4 \
  resume True \
  use_smpl_depth True \
  save_depth False \
  gpus "0,"
```

```
CUDA_VISIBLE_DEVICES=0 python run.py \
  --type evaluate \
  --cfg_file configs/thuman_no_smpl.yaml \
  virt_data_root data/THuman/val \
  ratio 0.5 \
  H 1024 W 1024 \
  run_mode test \
  test_mode model_x_motion_x \
  exp_name thuman_depth_no_smpl4 \
  resume True \
  test_sample_cam True \
  test.epoch 30 \
  exp_folder_name perceptual \
  use_smpl_depth True \
  save_depth True \
  gpus "0,"
```


# 11.18 new implementations

# debugs - multiple object training - working
```
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --cfg_file configs/thuman_no_smpl.yaml \
  virt_data_root data/THuman/train \
  H 1024 W 1024 \
  run_mode train \
  jitter True \
  exp_name thuman_depth_no_smpl5 \
  resume True \
  gpus "0,"
```

```
CUDA_VISIBLE_DEVICES=0 python run.py \
  --type evaluate \
  --cfg_file configs/thuman_no_smpl.yaml \
  virt_data_root data/THuman/val \
  ratio 0.5 \
  H 1024 W 1024 \
  run_mode test \
  test_mode model_x_motion_x \
  exp_name thuman_depth_no_smpl5 \
  resume True \
  test_sample_cam True \
  test.epoch 30 \
  exp_folder_name perceptual_30 \
  gpus "0,"
```

# debugs - thuman_depth_no_smpl6, full object dataset - smplx fill
```
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --cfg_file configs/thuman_no_smpl.yaml \
  virt_data_root data/THuman/train \
  H 1024 W 1024 \
  run_mode train \
  jitter True \
  exp_name thuman_depth_no_smpl6 \
  resume True \
  use_smpl_depth True \
  save_depth False \
  gpus "0,"
```

```
CUDA_VISIBLE_DEVICES=0 python run.py \
  --type evaluate \
  --cfg_file configs/thuman_no_smpl.yaml \
  virt_data_root data/THuman/val \
  ratio 0.5 \
  H 1024 W 1024 \
  run_mode test \
  test_mode model_x_motion_x \
  exp_name thuman_depth_no_smpl6 \
  resume True \
  test_sample_cam True \
  test.epoch -1 \
  exp_folder_name perceptual \
  use_smpl_depth True \
  save_depth True \
  gpus "0,"
```


# debugs - thuman_depth_no_smpl7, full object dataset - compare, no smplx fill
```
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --cfg_file configs/thuman_no_smpl.yaml \
  virt_data_root data/THuman/train \
  H 1024 W 1024 \
  run_mode train \
  jitter True \
  exp_name thuman_depth_no_smpl7 \
  resume True \
  use_smpl_depth False \
  gpus "0,"
```

```
CUDA_VISIBLE_DEVICES=0 python run.py \
  --type evaluate \
  --cfg_file configs/thuman_no_smpl.yaml \
  virt_data_root data/THuman/val \
  ratio 0.5 \
  H 1024 W 1024 \
  run_mode test \
  test_mode model_x_motion_x \
  exp_name thuman_depth_no_smpl7 \
  resume True \
  test_sample_cam True \
  test.epoch -1 \
  exp_folder_name perceptual \
  use_smpl_depth False \
  gpus "0,"
```



# full train dataset (train with obj 0004 to 0031 in previous)
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --cfg_file configs/thuman_no_smpl.yaml \
  virt_data_root data/THuman/train \
  H 1024 W 1024 \
  run_mode train \
  jitter False \
  exp_name thuman_depth_no_smpl_full_31 \
  resume True \
  use_smpl_depth True \
  save_depth False \
  gpus "0,"
```

```
CUDA_VISIBLE_DEVICES=0 python run.py \
  --type evaluate \
  --cfg_file configs/thuman_no_smpl.yaml \
  virt_data_root data/THuman/val \
  ratio 0.5 \
  H 1024 W 1024 \
  run_mode test \
  test_mode model_x_motion_x \
  exp_name thuman_depth_no_smpl_full_31 \
  resume True \
  test_sample_cam True \
  test.epoch 250 \
  exp_folder_name perceptual \
  use_smpl_depth True \
  save_depth True \
  gpus "0,"
```


# Run depth transformer
```
python train_depth_transformer.py --cfg_file configs/depth_transformer.yaml
python infer_depth_transformer.py --cfg_file configs/depth_transformer_test.yaml
```

# Run variance map
```
python get_feature_variance.py --cfg_file configs/variance_map.yaml virt_data_root data/THuman/train split train
python get_feature_variance.py --cfg_file configs/variance_map.yaml virt_data_root data/THuman/val split val
```