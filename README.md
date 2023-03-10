# Super Resolution

## About the Project
This project’s purpose was to evaluate the applicability of super resolution algorithms to a business process of Mattoboard, a startup company. Mattoboard (https://mattoboard.com/) builds 3D online material boards for interior designers. Some examples of the 3D material boards are shown below.
![material-board](https://github.com/sooolee/super-resolution/blob/main/images_readme/mattoboard-image.png?raw=true)



## Problem Statement
The types of images dealt with by Mattoboard encompass interior materials such as tiles, stones, woods, fabrics, carpets, wallpapers, metals, glasses, etc. The raw images of these materials are typically obtained from suppliers. Then the raw images need to be upscaled and converted into 3D PBR objects. This process is currently being done manually by hired designers. If any or all parts of this process can be done automatically, it could tremendously save them cost and time. I focused on the upscaling part of the process. 


## Review of Algorithms for Implementability
Various super resolution algorithms were reviewed through comparisons of the output quality, training codes accessibility, and quick inferences wherever possible. Models reviewed include [DATSR](https://arxiv.org/pdf/2207.11938.pdf), [EDSR](https://arxiv.org/pdf/1707.02921v1.pdf), [ISR](https://arxiv.org/pdf/1802.08797.pdf), [GLEAN](https://arxiv.org/pdf/2207.14812v1.pdf), [BAM](https://arxiv.org/pdf/2104.07566.pdf), [DASR](https://arxiv.org/pdf/2203.14216.pdf), and most importantly the SRGAN family ([SRGAN](https://arxiv.org/pdf/1609.04802.pdf), [ESRGAN](https://arxiv.org/pdf/1809.00219v2.pdf), [Real-ESRGAN](https://arxiv.org/pdf/2107.10833v2.pdf), and [A-ESRGAN](https://arxiv.org/pdf/2112.10046v1.pdf)). After visual inspections of the outputs and comparisons of PSNR or NIQE scores reported by the authors, DASR, Real-ESRGAN and A-ESRGAN made it to the final list. In terms of documentation for implementation, all these three models provided codes for training in their GitHub pages. After further review of the training codes and output quality, I decided to go with A-ESRGEN ([github](https://github.com/stroking-fishes-ml-corp/A-ESRGAN)) for further training, especially the Multi model where two multi scale U-net based discriminators were used instead of a single discriminator. 


## Platform - AWS EC2
Mattobaord provided me access to their AWS EC2 server, so all the training and inferences were made on the EC2 server. 

## Datasets
Various raw material images obtained from the web, typically from material suppliers’ webpages, were collected. This [full dataset](https://docs.google.com/spreadsheets/d/1McfQi_LJ9hh9d9V-cmnto6IzkvhZXh02FcEpwOfhkpk/edit#gid=2143057607) has 6,125 images of various materials. A reduced version of the same dataset was used in a training as well, where one-colored and no-pattern samples were removed. This [reduced dataset](https://docs.google.com/spreadsheets/d/1p50u6M0FAERrh01t0fEvLBQBa3gL8eDxdqLG9CRNcQM/edit#gid=2143057607) has 4,750 images. This resulted in two different trained models for comparison. 


## Implementation Strategies
First, before any fine-tuning or training, I made inferences using the three pre-trained models (Real-ESRGAN, A-ESRGAN-Single and A-ESRGAN-Multi models) to set the basis for comparisons. 

It was very hard to say one is better than the other purely based on visual inspections because the results varied depending on the types of materials (fabric vs. tile). 

While there are many good quality outputs, I am presenting some of the lower quality examples below to hightlight the weaknesses of the current models. 

**From Left to Right Columns: Original-Low Resolution >>> Real-ESRGAN >>> A-ESRGAN-Single >>> A_ESRGAN-Multi**

![initial_inferences](https://github.com/sooolee/super-resolution/blob/main/images_readme/initial_inferences.png?raw=true)

- One noticeable thing about the outputs of Real_ESRGAN was the change of the color tone. That is, most of the upscaled outputs slightly lost yellow tone and projected whiter tone. 
- For all models, fabrics and carpets seemed to be the toughest ones to upscale as most outputs failed capturing the texture. Some of them look much worse than its original low-resolution version. 

While the outputs of the A-ESRGAN-Multi scale model weren’t necessarily the best, according to the authors of A-ESRGAN-Multi, the multi scale discriminators model captures the texture better. So I chose ***A-ESRGAN-Multi*** for further training hoping it would help better capture the fabric and carpet textures.

Three training approaches were performed with the A-ESRGAN-Multi model. 

- Fine-tuning with the full dataset
- Training 1: Initialization of generator and training of A-ESRGAN-Multi using the full dataset
- Training 2: Initialization of generator using the full dataset and training of A-ESRGAN-Multi using the reduced dataset

Fine-tuning didn’t make much difference from the original pretrained model. So I moved on to full training.

According to the authors, A-ESRGAN is trained in two stages, which have the same data synthesis process and training pipeline, except for the loss functions. This training process is the same as for Real-ESRGAN.
1. First train Real-ESRNet with L1 loss from the pre-trained model ESRGAN.
2. Use the trained Real-ESRNet model as an initialization of the generator, and train the A-ESRGAN with a combination of L1 loss, perceptual loss and GAN loss. 

## Codes

### Download A-ESRGAN.git and install basicsr.

```
git clone https://github.com/stroking-fishes-ml-corp/A-ESRGAN.git
```
Under the newly created A-ESRGAN directory:
```
pip install basicsr
```
### Dataset preparation
Generate multi-scale images - downsample material images to obtain several Ground-Truth images with different scales
```
python scripts/generate_multiscale_DF2K.py --input datasets/materials --output datasets/materials_multiscale
```
Create a folder to store the meta information, which is created as the next step
```
mkdir -P ./datasets/meta-info
```
Prepare a txt for meta information
```
python scripts/generate_meta_info.py --input datasets/materials_multiscale --root datasets --meta_info datasets/meta-info/meta_info_materials_multiscale.txt
```

### Train Real-ESRNet (Initialization of Generator)

Download pre-trained model ESRGAN into experiments/pretrained_models
```
curl -L 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth' > ./experiments/pretrained_models/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth
```
Modify `train_realesrnet_x4plus.yml` file to set hyperparameters
```
vi options/train_realesrnet_x4plus.yml
```
Train with a single GPU in the debug mode
```
python train.py -opt options/train_realesrnet_x4plus.yml --debug
```
The formal training with the `--auto_resume` argument to automatically resume the training if necessary
```
python train.py -opt options/train_realesrnet_x4plus.yml --auto_resume
```
### Train A-ESRGAN

After the training of Real-ESRNet, I now have the file `./experiments/train_RealESRNetx4plus_1000k_B12G4_fromESRGAN/model/net_g_1000000.pth`

Download the latest A-ESRGAN Generator Model
```
curl -L 'https://github.com/aesrgan/A-ESRGAN/releases/download/v1.0.0/A_ESRGAN_Multi.pth' > ./experiments/pretrained_models/A_ESRGAN_Multi.pth
```
Download the latest A-ESRGAN Discriminator Model
```
curl -L 'https://github.com/stroking-fishes-ml-corp/A-ESRGAN/releases/download/v1.0.0/A_ESRGAN_Multi_D.pth' > ./experiments/pretrained_models/A_ESRGAN_Multi_D.pth
```
Now I have pretrained model discriminator `A_ESRGAN_Multi_D.pth` and generator `A_ESRGAN_Multi.pth`. Modify the yml file to set hyperparameters as needed. 

```
vi options/train_multiaesrgan_x4plus.yml
```
Train with a single GPU in the debug mode
```
python train.py -opt options/train_multiaesrgan_x4plus.yml --debug
```
The formal training with the `--auto_resume` argument to automatically resume the training if necessary
```
nohup python train.py -opt options/train_multiaesrgan_x4plus.yml --auto_resume
```


### Make Inferences
 Inferences were made using the following line. 

```
python inference_aesrgan.py --model_path=experiments/pretrained_models/net_g_150000.pth --input=inputs --output=outputs --tile=200
```

The training from scratch with the full material dataset (Training 1) improved the image qualities in all categories (image examples are provided later). But it seemed that the model still didn’t capture the texture of fabrics and carpets very well. I re-reviewed the datasets and removed one-colored and no-pattern samples as I thought these samples weren’t really helping the model to learn any textures or even patterns. Then the same training process was repeated with the reduced dataset (Training 2). 

## Training Hyperparameters

In addition to the different datasets, I made some changes to the hyperparameters for each training. Here is the summary of the hyperparameters used, compared to the original model by the authors.  

![hyperparameters](https://github.com/sooolee/super-resolution/blob/main/images_readme/hyperparameters.png?raw=true)


Following hyperparameters are worth noting as they would have impacted the output quality. Since no ablation study is performed, it is not possible to know how much each of these impacted the quality.

- ***Initialization of generator:*** Training of ESRNet was done only once but resulted in multiple checkpoints. Training 1 used the checkpoint at 35,000 iteration and Training 2 used the one at 55,000 iteration. The loss values of 55,000 were slightly better.
- ***Total iteration:*** I increased the total iteration for Training 2 as I lowered the learning rate. Retrospectively, I think the iteration should have been increased more to match the learning rate decrease. 
- ***Learning rate:*** Training 1 used a learning rate of 1e-4, same as the original study. A learning rate of 1e-5 was used for Training 2.
- ***Loss weight ratios between (Pixel : Perceptual : GAN):*** The original study used the same ratios between pixel and perceptual losses (1 : 1 : 0.1). I increased the pixel loss ratio to 2, and GAN loss to 0.2 (2 : 1 : 0.2). The rationale behind it is that since I’m dealing with interior material images, where the contents are mostly patterns, the perceptual losses had less meaning. For training 2, the ratio of (1 : 1 : 0.1) is used. 

> - I once trained the model with a (4 : 1 : 0.2) ratio and the model inferences generated a lot of artifacts, which is a known issue of the pixel loss focused models. 


## Results

Let's see the performace of the newly trained models: The same lower quality examples are presented below for comparisons. 

**From Left to Right Columns: Original-LR >>> A_ESRGAN-Multi >>> Training 1 >>> Training 2**
![trained_inferences](https://github.com/sooolee/super-resolution/blob/main/images_readme/trained_inferences.png?raw=true)

- Both Training 1 and 2 Models show improvements from the original A-ESRGAN-Multi inferences for most types of materials, and especially so for fabrics and carpets. 
- I was expecting better results for fabrics and carpets from Training 2 Model as the datasets were more focused on those. But interestingly, Training 1 Model outputs look better. I can’t tell whether it is because Training 1 used more pixel loss ratio than perceptual loss, or because Training 2 suffered from lack of iterations while having much lower learning rate. This is because I made changes to multiple hyperparameters all at once. My mistake!!! 
- For some types of materials, the outputs of the trained models weren’t necessarily better, just different. 
- Outputs by Training 1 Model shows slight loss of yellow tint in most of the samples. 

## Conclusions
This was a very fun project for me to learn through researching and training different algorithms with different hyperparameters and datasets. Unfortunately, the quality of outputs using the two trained models are not acceptable for Mattoboard. 

However, I retrospectively think there are some weaknesses in my training approach. First I had very small number of training dataset especially for training from scratch. In addition, the maximum total iteration number was only 11 epochs. Not enough for pretraining the model. I wonder as well why fine-tuning didn't work well as I think this could have been the ideal case for transfer learning. 

This leaves that there still should be good opportunities of making AI upscaling algorithms to work at a commercial level. I plan to come back to this project and try some more later. 

***Lessons learned:*** Training with different hyperparameters needs to be more carefully designed in the beginning so that impacts by each can be explained. 

## {UPDATE} -- Stable Diffusion Upscaler
I tried Stable Diffusion Upscaler `stabilityai/stable-diffusion-x4-upscaler` and am very impressed by the results. The Stable Diffusion outputs for some of the examples from above are presented below. The text propmts were left blank for all samples intentionally.

**From Left to Right Columns: Original-LR >>> Training 1 Model >>> Stable-Diffusion**
![sd_examples](https://github.com/sooolee/super-resolution/blob/main/images_readme/sd_inferences.png?raw=true)

This is very exciting and promising for Mattoboard and I will look further into this.
