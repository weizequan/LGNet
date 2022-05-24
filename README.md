# LGNet

Image Inpainting With Local and Global Refinement

## Prerequisites
- Python 3
- NVIDIA GPU + CUDA cuDNN
- PyTorch 1.3.1

## Run
1. train the model
```
python train.py --dataroot no_use --name celebahq_LGNet --model pix2pixglg --netG1 unet_256 --netG2 resnet_4blocks --netG3 unet256 --netD snpatch --gan_mode lsgan --input_nc 4 --no_dropout --direction AtoB --display_id 0 --gpu_ids 0
```
2. test the model
```
python test_and_save.py --dataroot no_use --name celebahq_LGNet --model pix2pixglg --netG1 unet_256 --netG2 resnet_4blocks --netG3 unet256 --gan_mode nogan --input_nc 4 --no_dropout --direction AtoB --gpu_ids 0
```

## Pretrained Models
You can download the pretrained model from [Celeba-HQ](https://drive.google.com/drive/folders/1waZDA4-ubmZXGjkd_FQIAx-gb0Hd76bI?usp=sharing). Then put them into the ./checkpoints/celebahq_LGNet/.

## Citation
If you find this useful for your research, please use the following.

```
@ARTICLE{9730792,
  author={Quan, Weize and Zhang, Ruisong and Zhang, Yong and Li, Zhifeng and Wang, Jue and Yan, Dong-Ming},
  journal={IEEE Transactions on Image Processing}, 
  title={Image Inpainting With Local and Global Refinement}, 
  year={2022},
  volume={31},
  pages={2405-2420}
}
```

## Acknowledgments
This code borrows from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [RFR](https://github.com/jingyuanli001/RFR-Inpainting).
