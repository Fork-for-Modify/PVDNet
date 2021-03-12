# PVDNet: Pixel Volume based Video Debluring Network
![License CC BY-NC](https://img.shields.io/badge/license-GNU_AGPv3-blue.svg?style=plastic)

<p align="center">
   <img src="./assets/network.jpg" />
</p>

This repository contains the official PyTorch implementation of the following paper:

> **[Recurrent Video Deblurring with Blur-Invariant Motion Estimation and Pixel Volumes (TODO)]()**<br>
> Hyeongseok Son, Junyong Lee, Jonghyeop Lee, Sunghyun Cho, Seungyong Lee, TOG 2021

## About the Research
<details>
    <summary><i>Click here</i></summary>
    <h3> Overall Framework </h3>
    <p align="center">
    <img width=50% src="./assets/framework.jpg" />
    </p>
    <p>
        Our video deblurring framework consists of three modules: a blur-invariant motion estimation network (<i>BIMNet</i>), a pixel volume generator, and a pixel volume-based deblurring network (<i>PVDNet</i>).
        We first train *BIMNet*; after it has converged, we combine the two networks with the pixel volume generator.
        We then fix the parameters of <i>BIMNet</i> and train <i>PVDNet</i> by training the entire network.
    </p>
    <h3> Blur-Invariant Motion Estimation Network (<i>BIMNet</i>)</h3>
    <p>
        To estimate motion between frames accurately, we adopt <a "https://arxiv.org/pdf/1805.07036.pdf">LiteFlowNet</a> and train it with a blur-invariant loss so that the trained network can estimate blur-invariant optical flow between frames.
        We train *BIMNet* with ablur-invariant loss <img src="https://latex.codecogs.com/svg.latex?L_{BIM}^{\alpha\beta}" />, which is defined as (refer Eq. 1 in the main paper):
    <p>
    <p align="center">
        <img src="./assets/BIMNet_eq.svg" />
    </p>
    <p align="center">
        <img width=80% src="./assets/BIMNet_figure.jpg" />
    </p>
    <p>
        The figure shows a qualitative comparison of different optical flow methods.
        The results of the other methods contain severely distorted structures due to errors in their optical flow maps.
        In contrast, the results of BIMNets show much less distortions.
    </p>
    <h3> Pixel Volume for Motion Compensation </h3>
    <p>
        We propose a novel pixel volume that provides multiple candidates for matching pixels between images.
        Moreover, a pixel volume provides an additional cue for motion compensation based on the majority.
    </p>
    <p align="center">
        <img width=60% src="./assets/PV.jpg" />
    </p>
    <p>
        Our pixel volume approach leads to the performance improvement of video deblurring by utilizing the multiple candidates in a pixel volume in two aspects: 1) in most cases, the majority cue for the correct match would help as the statistics (Sec. 4.4 in the main paper) shows, and 2) in other cases, <i>PVDNet</i> would exploit multiple candidates to estimate the correct match referring to nearby pixels with majority cues.
    </P>
</details>


## Getting Started
### Prerequisites
*Tested environment*

![Ubuntu18.04](https://img.shields.io/badge/Ubuntu-16.0.4%20&%2018.0.4-blue.svg?style=plastic)
![Python 3.8.8](https://img.shields.io/badge/Python-3.8.8-green.svg?style=plastic)
![PyTorch 1.8.0](https://img.shields.io/badge/PyTorch-1.8.0-green.svg?style=plastic)
![CUDA 10.2](https://img.shields.io/badge/CUDA-10.2%20&%2011.1-green.svg?style=plastic)

1. **Install requirements** 
    * `pip install -r requirements.txt`

        > **Note:**        
        >
        > If you face a problem with `cupy` during installation, modify `cupy-cuda1xx` in `requirements.txt` to indicate proper version of the CUDAToolkit when PyTorch is installed (*e.g.*, `cupy-cuda102`).

2. **Datasets**
    * Download and unzip [Su *et al.*'s dataset](https://www.dropbox.com/s/8daduee9igqx5cw/DVD.zip?dl=0) and [Nah *et al.*'s dataset](https://www.dropbox.com/s/5ese6qtbwy7fsoh/nah.zip?dl=0) under `[DATASET_ROOT]`:

        ```
        ├── [DATASET_ROOT]
        │   ├── train_DVD
        │   ├── test_DVD
        │   ├── train_nah
        │   ├── test_nah
        ```

        > **Note:**
        >
        > `[DATASET_ROOT]` is currently set to `./datasets`. It can be specified by modifying `config.data_offset` in `./configs/config.py`.

3. **Pre-trained models**
    * Download and unzip [pretrained weights](https://www.dropbox.com/sh/frpegu68s0yx8n9/AACrptFFhxejSyKJBvLdk9IJa?dl=0) under `./ckpt/`:

        ```
        ├── ./ckpt
        │   ├── BIMNet.pytorch
        │   ├── PVDNet_DVD.pytorch
        │   ├── PVDNet_nah.pytorch
        │   ├── PVDNet_large_nah.pytorch
        ```

### Logs
* Training and tesing logs will be saved under `[LOG_ROOT]/PVDNet_TOG2021/[mode]/`:

    ```
    ├── [LOG_ROOT]
    │   ├── PVDNet_TOG2021
    │   │   ├── [mode]
    │   │   │   ├── checkpoint      # model checkpoint and resume states
    │   │   │   ├── log             # scalar/image log for tensorboard
    │   │   │   ├── sample          # sample images of training and validation
    │   │   │   ├── config          # config file
    │   │   │   ├── result          # results images of evaluation
    │   │   │   ├── cost.txt        # network size and MACs measured on an image with the size of (1, 3, 1280, 720)
    │   │   │   ├── [network].py    # network file
    ```

* In `./config/config.py`, you may configure following items:
    * `config.log_offset`: configures `[LOG_ROOT]`. Default: `./logs`
    * `config.write_ckpt_every_epoch`: configures an epoch period for saving checkpoints, resume states and scalar logs. Default: 4
    * `config.write_log_every_itr`: configures an iteration period for saving sample images. Default: `{'train':16, 'valid':16}`
    * `config.refresh_image_log_every_epoch`: configures an epoch period for erasing sample images. Defualt: `{'train':65, 'valid': 20}`

## Testing models of TOG2021

```shell
## Table 4 in the main paper (Evaluation on Su etal's dataset)
# Our final model 
python run.py --mode PVDNet_DVD --config config_PVDNet --data DVD --ckpt_abs_name ckpt/PVDNet_DVD.pytorch

## Table 5 in the main paper (Evaluation on Nah etal's dataset)
# Our final model 
python run.py --mode PVDNet_nah --config config_PVDNet --data nah --ckpt_abs_name ckpt/PVDNet_nah.pytorch

# Larger model
python run.py --mode PVDNet_large_nah --config config_PVDNet_large --data nah --ckpt_abs_name ckpt/PVDNet_large_nah.pytorch
```

> **Note:**
>
> Testing results will be saved in `[LOG_ROOT]/PVDNet_TOG2021/[mode]/result/quanti_quali/[mode]_[epoch]/[data]/`.

* options
    * `--data`: The name of a dataset to evaluate: `DVD` | `nah` | `random`. Default: `DVD`
        * The data structure can be modified in the function `set_eval_path(..)` in `./configs/config.py`.
        * `random` is for testing models with any video frames, which should be placed as `[DATASET_ROOT]/random/[video_name]/*.[jpg|png]`. 

## Training & testing the network
### Training

```shell
# multi GPU (with DistributedDataParallel) example
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B -m torch.distributed.launch --nproc_per_node=4 --master_port=9000 run.py \
            --is_train \
            --mode PVDNet_DVD \
            --config config_PVDNet \
            --trainer trainer \
            --data DVD \
            -LRS CA \
            -b 2 \
            -th 8 \
            -dl \
            -ss \
            -dist

# resuming example (trainer will load checkpoint saved at 100 epoch, training resumes form 101 epoch)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B -m torch.distributed.launch --nproc_per_node=4 --master_port=9000 run.py \
            ... \
            -th 8 \
            -r 100 \
            -ss \
            -dist

# single GPU (with DataParallel) example
CUDA_VISIBLE_DEVICES=0 python -B run.py \
            --is_train \
            --mode PVDNet_DVD \
            --config config_PVDNet \
            --trainer trainer \
            --data DVD \
            -LRS CA \
            -b 8 \
            -th 8 \
            -dl \
            -ss
```
* Options
    * `--is_train`: If it is specified, `run.py` will train the network. Default: `False`
    * `--mode`: The name of a model to train. The logging folder named with the `[mode]` will be created as `[LOG_ROOT]/PVDNet_TOG2021/[mode]/`. Default: `PVDNet_DVD`
    * `--config`: The name of config file located as in `./config/[config].py`. Default: `None`, and the default should not be changed.
    * `--trainer`: The name of trainer file located as `./models/trainers/[trainer].py`. Default: `trainer`
    * `--data`: The name of dataset: `DVD` | `nah`. Default: `DVD`
    * `--network`: The name of network file (of PVDNet) located as `./models/archs/[network].py`. Default: `PVDNet`
    * `-LRS`: Learning rate scheduler for training: `CA`(Cosine annealing scheduler) | `LD`(step decay schedule). Default: `CA`
    * `-b`, `--batch_size`: The batch size. For the multi GPU (`DistributedDataParallel`), the total batch size will be, `nproc_per_node * b`. Default: 8
    * `-th`, `--thread_num`: The number of thread (`num_workers`) used for the data loader. Default: 8
    * `-dl`, `--delete_log`: The option whether to delete logs under `[mode]` (i.e., `[LOG_ROOT]/PVDNet_TOG2021/[mode]/*`). Option works only when `--is_train` is specified. Default: `False`
    * `-r`, `--resume`: Resume training with specified epoch # (e.g., `-r 100`). Note that `-dl` should not be specified with this option.
    * `-ss`, `--save_sample`: Save sample images for both training and testing. Images will be saved in `[LOG_ROOT]/PVDNet_TOG2022/[mode]/sample/`. Default: `False`
    * `-dist`: Enables multi-processing with `DistributedDataParallel`. Default: `False`

<!--
<details>
    <summary>Otions</summary>
    <ul>
        <li><code>--is_train</code>: If it is specified, <code>run.py</code> will train the network. Default: <code>False</code></li>
        <li><code>--mode</code>: The name of a model to train. The logging folder named with the <code>[mode]</code> will be created as <code>[LOG_ROOT]/PVDNet_TOG2021/[mode]/</code>. Default: <code>PVDNet_DVD</code></li>
        <li><code>--config</code>: The name of config file located as in <code>./config/[config].py</code>. Default: <code>None</code>, and the default should not be changed.</li>
        <li><code>--trainer</code>: The name of trainer file located as </code>./models/trainers/[trainer].py</code>. Default: <code>trainer</code></li>
        <li><code>--data</code>: The name of dataset: <code>DVD</code> | <code>nah</code>. Default: <code>DVD</code></li>
        <li><code>--network</code>: The name of network file (of PVDNet) located as <code>./models/archs/[network].py</code>. Default: <code>PVDNet</code></li>
        <li><code>-LRS</code>: Learning rate scheduler for training: <code>CA</code>(Cosine annealing scheduler) | <code>LD</code>(step decay schedule). Default: <code>CA</code></li>
        <li><code>-b</code>, <code>--batch_size</code>: The batch size. For the multi GPU (<code>DistributedDataParallel</code>), the total batch size will be, <code>nproc_per_node <li>b</code>. Default: 8</li>
        <li><code>-th</code>, <code>--thread_num</code>: The number of thread (<code>num_workers</code>) used for the data loader. Default: 8</li>
        <li><code>-dl</code>, <code>--delete_log</code>: The option whether to delete logs under <code>[mode]</code> (i.e., <code>[LOG_ROOT]/PVDNet_TOG2021/[mode]/*</code>). Option works only when <code>--is_train</code> is specified. Default: <code>False</code></li>
        <li><code>-r</code>, <code>--resume</code>: Resume training with specified epoch # (e.g., <code>-r 100</code>). Note that <code>-dl</code> should not be specified with this option.</li>
        <li><code>-ss</code>, <code>--save_sample</code>: Save sample images for both training and testing. Images will be saved in <code>[LOG_ROOT]/PVDNet_TOG2022/[mode]/sample/</code>. Default: <code>False</code></li>
        <li><code>-dist</code>: Enables multi-processing with <code>DistributedDataParallel</code>. Default: <code>False</code></li>
    </ul>
</details>
-->

### Testing

```shell
python run.py --mode [mode] --data [DATASET]
# e.g., python run.py --mode PVDNet_DVD --data DVD
```

> **Note:**
>
> * Specify only `[mode]` of the trained model. `[config]` doesn't have to be specified, as it will be automatically loaded.
>
> * Testing results will be saved in `[LOG_ROOT]/PVDNet_TOG2021/[mode]/result/quanti_quali/[mode]_[epoch]/[data]/`.

* Options
    * `--mode`: The name of a model to test.
    * `--data`: The name of a dataset to evaluate: `DVD` | `nah` | `random`. Default: `DVD` 
        *  The data structure can be modified in the function `set_eval_path(..)` in `./configs/config.py`.
        * `random` is for testing models with any video frames, which should be placed as `[DATASET_ROOT]/random/[video_name]/*.[jpg|png]`.
    * `-ckpt_name`: Load the checkpoint with the name of the checkpoint under `[LOG_ROOT]/PVDNet_TOG2021/[mode]/checkpoint/train/epoch/ckpt/` (e.g., `python run.py --mode PVDNet_DVD --data DVD --ckpt_name PVDNet_DVD_00100.pytorch`).
    * `-ckpt_abs_name`. Loads the checkpoint of the absolute path (e.g., `python run.py --mode PVDNet_DVD --data DVD --ckpt_abs_name ./ckpt/PVDNet_DVD.pytorch`).
    * `-ckpt_epoch`: Loads the checkpoint of the specified epoch (e.g., `python run.py --mode PVDNet_DVD --data DVD --ckpt_epoch 100`). 
    * `-ckpt_sc`: Loads the checkpoint with the best validation score (e.g., `python run.py --mode PVDNet_DVD --data DVD --ckpt_sc`).

## Citation
If you find this code useful, please consider citing:
```
@artical{Son_2021_TOG,
    author = {Son, Hyeongseok and Lee, Junyong and Lee, Jonghyeop and Cho, Sunghyun and Lee, Seungyong},
    title = {Recurrent Video Deblurring with Blur-Invariant Motion Estimation and Pixel Volumes},
    journal = {ACM Transactions on Graphics},
    year = {2021}
}
```

## Contact
Open an issue for any inquiries.
You may also have contact with [sonhs@postech.ac.kr](mailto:sonhs@postech.ac.kr) or [junyonglee@postech.ac.kr](mailto:junyonglee@postech.ac.kr)

## Resources
All material related to our paper is available by following links:

| Link |
| :-------------- |
| [The main paper (todo)](https://drive.google.com/file/d/1mRVo3JefkgRd2VdJvG5M-8xWtvl60ZWg/view?usp=sharing) |
| [Supplementary Files (todo)](https://drive.google.com/file/d/1sQTGHEcko2HxoIvneyrot3bUabPrN5l1/view?usp=sharing) |
| [Checkpoint Files](https://www.dropbox.com/sh/frpegu68s0yx8n9/AACrptFFhxejSyKJBvLdk9IJa?dl=0) |
| [Su *et al* [2017]'s dataset](https://www.dropbox.com/s/8daduee9igqx5cw/DVD.zip?dl=0) ([reference](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/#dataset)) |
| [Nah *et al.* [2017]'s dataset](https://www.dropbox.com/s/5ese6qtbwy7fsoh/nah.zip?dl=0) ([reference](https://seungjunnah.github.io/Datasets/gopro)) |

## License
This software is being made available under the terms in the [LICENSE](LICENSE) file.

Any exemptions to these terms require a license from the Pohang University of Science and Technology.

## About Coupe Project
Project ‘COUPE’ aims to develop software that evaluates and improves the quality of images and videos based on big visual data. To achieve the goal, we extract sharpness, color, composition features from images and develop technologies for restoring and improving by using them. In addition, personalization technology through user reference analysis is under study.  
    
Please checkout other Coupe repositories in our [Posgraph](https://github.com/posgraph) github organization.

### Useful Links
* [Coupe Library](http://coupe.postech.ac.kr/)
* [POSTECH CG Lab.](http://cg.postech.ac.kr/)
