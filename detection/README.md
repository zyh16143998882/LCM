# LCM: Locally Constrained Compact Point Cloud Model for Masked Point Modeling

Our detection implementation of Point-MAE w/ LCM builds upon the [**3DETR codebase**](https://github.com/facebookresearch/3detr). For details on environment setup and data configuration, please refer to the 3DETR documentation.

## Training (e.g. Point-MAE (w/ LCM))
To pre-train Point-MAE (w/ LCM) on ShapeNet, you can run the following command: 

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset_name scannet --checkpoint_dir lcm_mae --model_name 3detr_lcm --loss_giou_weight 1 --loss_no_object_weight 0.25 --nqueries 256 --ngpus 4 --batchsize_per_gpu 8 --enc_dim 384 --pretrain_ckpt ../ckpts/pretrained/lcm_mae.pth
```