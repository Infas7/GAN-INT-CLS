# Text-to-Image-Synthesis

## Intoduction

This is a pytorch implementation of [Generative Adversarial Text-to-Image Synthesis paper](https://arxiv.org/abs/1605.05396), we train a conditional generative adversarial network, conditioned on text descriptions, to generate images that correspond to the description. The network architecture is shown below (Image from [1]). This architecture is based on DCGAN.


## Datasets

We used [Caltech-UCSD Birds 200](http://www.vision.caltech.edu/visipedia/CUB-200.html) and [Flowers](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) datasets, we converted each dataset (images, text embeddings) to hd5 format.

**Hd5 file taxonomy**
`

- split (train | valid | test )
  - example_name
    - 'name'
    - 'img'
    - 'embeddings'
    - 'class'
    - 'txt'

## Usage

### Prerequisites

Python 3.10

```
pip install -r requirements.txt
```

### Datasets

1. Download and extract the [birds](https://drive.google.com/file/d/0B0ywwgffWnLLLUc2WHYzM0Q2eWc/view?usp=sharing) and [flowers](https://drive.google.com/file/d/0B0ywwgffWnLLMl9uOU91MV80cVU/view?usp=sharing) and [COCO](https://drive.google.com/open?id=0B0ywwgffWnLLamltREhDRjlaT3M) caption data in Torch format.
2. Download and extract the [birds](https://drive.google.com/open?id=0B0ywwgffWnLLZW9uVHNjb2JmNlE) and [flowers](https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/view?resourcekey=0-Av8zFbeDDvNcF1sSjDR32w) and [COCO]() Text encoding.
3. Download and extract the [birds](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1) and [flowers](ww) and [COCO]() image data.
4. Use [convert_cub_to_hd5_script](convert_cub_to_hd5_script.py) or [convert_flowers_to_hd5_script](convert_flowers_to_hd5_script.py) script to convert the dataset.

### Training

`python runtime.py --ds

**Arguments:**

- `type` : GAN archiecture to use `(gan | wgan | vanilla_gan | vanilla_wgan)`. default = `gan`. Vanilla mean not conditional
- `dataset`: Dataset to use `(birds | flowers)`. default = `flowers`
- `split` : An integer indicating which split to use `(0 : train | 1: valid | 2: test)`. default = `0`
- `lr` : The learning rate. default = `0.0002`
- `diter` : Only for WGAN, number of iteration for discriminator for each iteration of the generator. default = `5`
- `save_path` : Path for saving the models.
- `l1_coef` : L1 loss coefficient in the generator loss fucntion for gan and vanilla_gan. default=`50`
- `l2_coef` : Feature matching coefficient in the generator loss fucntion for gan and vanilla_gan. default=`100`
- `pre_trained_disc` : Discriminator pre-tranined model path used for intializing training.
- `pre_trained_gen` Generator pre-tranined model path used for intializing training.
- `batch_size`: Batch size. default = `64`
- `num_workers`: Number of dataloader workers used for fetching data. default = `8`
- `epochs` : Number of training epochs. default=`200`
- `eval_batch_size` : Evaluation batch size. default = `512`
- `eval_interval` : Interval for evaluating GAN. default = `10`
- `ds` : Enable efficient data selection. default `False`
