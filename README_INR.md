# PVDNet for INR projects

> By zzh

> more info: https://github.com/codeslake/PVDNet/wiki/Training-&-Testing-Details

## train:
### multi GPU (with DistributedDataParallel) example
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B -m torch.distributed.launch --nproc_per_node=4 --master_port=9000 run.py \
            --is_train \
            --mode PVDNet_nah \
            --config config_PVDNet_inr \
            --trainer trainer \
            --data nah \
            -LRS CA \
            -b 2 \
            -th 8 \
            -ss \
            -dl \
            -dist

`CUDA_VISIBLE_DEVICES=0,1,2,3 python -B -m torch.distributed.launch --nproc_per_node=4 --master_port=9000 run.py --is_train  --mode PVDNet_nah --config config_PVDNet_inr --trainer trainer  --data nah -LRS CA -b 2 -th 8 -ss -dl -dist`

### resuming example (trainer will load checkpoint saved at 100 epoch, training will resume form 101 epoch)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B -m torch.distributed.launch --nproc_per_node=4 --master_port=9000 run.py \
            ... \
            -th 8 \
            -r 100 \
            -ss \
            -dist

### single GPU (with DataParallel) example
CUDA_VISIBLE_DEVICES=0 python -B run.py \
            ... \
            -ss

## test:
CUDA_VISIBLE_DEVICES=0 python run.py --mode PVDNet_nah --config config_PVDNet_inr --data nah --ckpt_abs_name logs/PVDNet_TOG2021/PVDNet_nah/checkpoint/train/epoch/ckpt/PVDNet_nah_00140.pytorch
